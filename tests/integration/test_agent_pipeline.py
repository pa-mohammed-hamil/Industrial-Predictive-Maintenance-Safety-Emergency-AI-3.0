
"""
INDUSTRA AI
tests/test_agent_pipeline.py

Industrial Multi-Agent Pipeline Test Suite

Run:
    pip install pytest pytest-asyncio
    pytest tests/test_agent_pipeline.py -v

Coverage:
    - Supervisor orchestration
    - Agent routing
    - Predictive maintenance
    - Anomaly detection
    - RUL prediction
    - Safety monitoring
    - Emergency escalation
    - Cost intelligence
    - Industrial RAG
    - Tool execution
    - Security guardrails
    - Agent evaluation
    - Pipeline reliability

This file uses a reference in-memory pipeline adapter.
Replace ReferenceAgentPipeline with the actual application adapter
for integration testing.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ============================================================
# DOMAIN MODELS
# ============================================================

class AgentType(str, Enum):
    SUPERVISOR = "supervisor"
    MAINTENANCE = "maintenance"
    ANOMALY = "anomaly"
    RUL = "rul"
    SAFETY = "safety"
    EMERGENCY = "emergency"
    COST = "cost"
    RAG = "rag"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Asset:
    asset_id: str
    name: str
    plant: str
    health_score: float
    temperature: float
    vibration: float
    pressure: float
    current: float
    rul_months: float
    failure_probability: float


@dataclass
class AgentRequest:
    query: str
    asset_id: str
    user_id: str = "test-user"
    tenant_id: str = "tenant-a"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    agent: AgentType
    success: bool
    data: Dict[str, Any]
    confidence: float = 1.0
    citations: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    success: bool
    asset_id: str
    selected_agents: List[AgentType]
    results: List[AgentResult]
    final_response: Dict[str, Any]
    errors: List[str] = field(default_factory=list)


# ============================================================
# SAMPLE INDUSTRIAL ASSETS
# ============================================================

@pytest.fixture
def assets() -> Dict[str, Asset]:
    return {
        "MTR-001": Asset(
            asset_id="MTR-001",
            name="Main Drive Motor",
            plant="Plant A",
            health_score=74,
            temperature=72,
            vibration=4.1,
            pressure=3.2,
            current=112,
            rul_months=48,
            failure_probability=0.32,
        ),
        "GEN-001": Asset(
            asset_id="GEN-001",
            name="Power Generator",
            plant="Plant B",
            health_score=52,
            temperature=91,
            vibration=6.8,
            pressure=5.2,
            current=185,
            rul_months=24,
            failure_probability=0.43,
        ),
        "PMP-001": Asset(
            asset_id="PMP-001",
            name="Cooling Pump",
            plant="Plant B",
            health_score=84,
            temperature=58,
            vibration=2.9,
            pressure=5.6,
            current=63,
            rul_months=69,
            failure_probability=0.20,
        ),
    }


# ============================================================
# REFERENCE AGENT PIPELINE
# ============================================================

class ReferenceAgentPipeline:
    """
    Deterministic reference implementation for testing.

    This class is NOT a trained AI model or production
    industrial control system.
    """

    def __init__(self, assets: Dict[str, Asset]):
        self.assets = assets
        self.execution_log: List[str] = []

        self.thresholds = {
            "temperature": 85,
            "vibration": 5.5,
            "pressure": 9,
            "current": 160,
        }

        self.knowledge_base = {
            "motor-maintenance": {
                "text": "Inspect bearings, lubrication and vibration.",
                "source": "manual://motor-maintenance",
            },
            "generator-safety": {
                "text": "Escalate abnormal generator temperature "
                        "and vibration to qualified personnel.",
                "source": "manual://generator-safety",
            },
        }

    def validate_request(self, request: AgentRequest) -> None:
        if not request.query.strip():
            raise ValueError("Query cannot be empty")

        if request.asset_id not in self.assets:
            raise KeyError("Unknown asset")

        if not request.tenant_id.strip():
            raise PermissionError("Tenant ID required")

        if len(request.query) > 5000:
            raise ValueError("Query exceeds maximum length")

    def route(self, query: str) -> List[AgentType]:
        q = query.lower()

        selected = []

        routing_rules = {
            AgentType.MAINTENANCE: [
                "maintenance", "repair", "inspection",
            ],
            AgentType.ANOMALY: [
                "anomaly", "abnormal", "sensor",
            ],
            AgentType.RUL: [
                "rul", "remaining useful life", "lifespan",
            ],
            AgentType.SAFETY: [
                "safety", "hazard", "risk",
            ],
            AgentType.EMERGENCY: [
                "emergency", "fire", "smoke", "overheat",
            ],
            AgentType.COST: [
                "cost", "expense", "replacement", "roi",
            ],
            AgentType.RAG: [
                "manual", "document", "procedure", "guideline",
            ],
        }

        for agent, keywords in routing_rules.items():
            if any(keyword in q for keyword in keywords):
                selected.append(agent)

        if not selected:
            selected.append(AgentType.MAINTENANCE)

        return selected

    async def execute_agent(
        self,
        agent_type: AgentType,
        asset: Asset,
    ) -> AgentResult:

        self.execution_log.append(agent_type.value)

        if agent_type == AgentType.MAINTENANCE:
            priority = (
                "high" if asset.health_score < 65
                else "medium" if asset.health_score < 80
                else "low"
            )

            return AgentResult(
                agent=agent_type,
                success=True,
                data={
                    "asset_id": asset.asset_id,
                    "priority": priority,
                    "recommendation": "Schedule condition inspection",
                },
                confidence=0.90,
            )

        if agent_type == AgentType.ANOMALY:
            anomalies = []

            for metric, threshold in self.thresholds.items():
                if getattr(asset, metric) > threshold:
                    anomalies.append(metric)

            return AgentResult(
                agent=agent_type,
                success=True,
                data={
                    "anomaly_detected": bool(anomalies),
                    "anomalous_sensors": anomalies,
                },
                confidence=0.95,
            )

        if agent_type == AgentType.RUL:
            return AgentResult(
                agent=agent_type,
                success=True,
                data={
                    "rul_months": asset.rul_months,
                    "failure_probability": asset.failure_probability,
                },
                confidence=0.85,
            )

        if agent_type == AgentType.SAFETY:
            critical = (
                asset.temperature > self.thresholds["temperature"]
                or asset.vibration > self.thresholds["vibration"]
            )

            return AgentResult(
                agent=agent_type,
                success=True,
                data={
                    "hazard_detected": critical,
                    "severity": (
                        Severity.HIGH.value
                        if critical else Severity.LOW.value
                    ),
                },
                confidence=0.93,
            )

        if agent_type == AgentType.EMERGENCY:
            emergency = (
                asset.temperature > 100
                or asset.vibration > 9
            )

            return AgentResult(
                agent=agent_type,
                success=True,
                data={
                    "emergency_detected": emergency,
                    "escalation_required": emergency,
                    "automatic_shutdown": False,
                },
                confidence=0.96,
            )

        if agent_type == AgentType.COST:
            expected_failure_cost = (
                asset.failure_probability * 250000
            )

            return AgentResult(
                agent=agent_type,
                success=True,
                data={
                    "expected_failure_cost": expected_failure_cost,
                    "currency": "INR",
                },
                confidence=1.0,
            )

        if agent_type == AgentType.RAG:
            document = (
                self.knowledge_base["generator-safety"]
                if asset.asset_id.startswith("GEN")
                else self.knowledge_base["motor-maintenance"]
            )

            return AgentResult(
                agent=agent_type,
                success=True,
                data={"answer": document["text"]},
                citations=[document["source"]],
                confidence=0.88,
            )

        return AgentResult(
            agent=agent_type,
            success=False,
            data={},
            errors=["Unsupported agent"],
            confidence=0.0,
        )

    async def run(
        self,
        request: AgentRequest,
    ) -> PipelineResult:

        self.validate_request(request)

        asset = self.assets[request.asset_id]
        agents = self.route(request.query)

        results = await asyncio.gather(
            *[
                self.execute_agent(agent, asset)
                for agent in agents
            ],
            return_exceptions=True,
        )

        valid_results: List[AgentResult] = []
        errors: List[str] = []

        for result in results:
            if isinstance(result, BaseException):
                errors.append(str(result))
            else:
                valid_results.append(result)

        final_response = {
            "asset_id": asset.asset_id,
            "asset_name": asset.name,
            "agent_outputs": {
                result.agent.value: result.data
                for result in valid_results
            },
            "citations": [
                citation
                for result in valid_results
                for citation in result.citations
            ],
        }

        return PipelineResult(
            success=(
                len(errors) == 0
                and all(r.success for r in valid_results)
                and len(valid_results) > 0
            ),
            asset_id=asset.asset_id,
            selected_agents=agents,
            results=valid_results,
            final_response=final_response,
            errors=errors,
        )


@pytest.fixture
def pipeline(assets):
    return ReferenceAgentPipeline(assets)


# ============================================================
# 1. SUPERVISOR & ROUTING TESTS
# ============================================================

def test_supervisor_routes_maintenance(pipeline):
    agents = pipeline.route("Predict maintenance requirements")
    assert AgentType.MAINTENANCE in agents


def test_supervisor_routes_anomaly(pipeline):
    agents = pipeline.route("Detect sensor anomaly")
    assert AgentType.ANOMALY in agents


def test_supervisor_routes_rul(pipeline):
    agents = pipeline.route("Estimate remaining useful life")
    assert AgentType.RUL in agents


def test_supervisor_routes_safety(pipeline):
    agents = pipeline.route("Assess equipment safety")
    assert AgentType.SAFETY in agents


def test_supervisor_routes_emergency(pipeline):
    agents = pipeline.route("Emergency overheating assessment")
    assert AgentType.EMERGENCY in agents


def test_supervisor_routes_cost(pipeline):
    agents = pipeline.route("Calculate replacement cost")
    assert AgentType.COST in agents


def test_supervisor_routes_rag(pipeline):
    agents = pipeline.route("Search equipment manual")
    assert AgentType.RAG in agents


def test_supervisor_routes_multiple_agents(pipeline):
    agents = pipeline.route(
        "Check safety, maintenance, RUL and cost"
    )

    assert AgentType.SAFETY in agents
    assert AgentType.MAINTENANCE in agents
    assert AgentType.RUL in agents
    assert AgentType.COST in agents


def test_supervisor_default_agent(pipeline):
    agents = pipeline.route("Analyze this equipment")
    assert agents == [AgentType.MAINTENANCE]


def test_supervisor_no_duplicate_agents(pipeline):
    agents = pipeline.route("maintenance repair inspection")
    assert len(agents) == len(set(agents))


# ============================================================
# 2. REQUEST VALIDATION TESTS
# ============================================================

def test_reject_empty_query(pipeline):
    with pytest.raises(ValueError):
        pipeline.validate_request(
            AgentRequest(query="", asset_id="MTR-001")
        )


def test_reject_unknown_asset(pipeline):
    with pytest.raises(KeyError):
        pipeline.validate_request(
            AgentRequest(query="Analyze", asset_id="UNKNOWN")
        )


def test_reject_missing_tenant(pipeline):
    with pytest.raises(PermissionError):
        pipeline.validate_request(
            AgentRequest(
                query="Analyze",
                asset_id="MTR-001",
                tenant_id="",
            )
        )


def test_reject_oversized_query(pipeline):
    with pytest.raises(ValueError):
        pipeline.validate_request(
            AgentRequest(
                query="x" * 5001,
                asset_id="MTR-001",
            )
        )


def test_accept_valid_request(pipeline):
    pipeline.validate_request(
        AgentRequest(
            query="Check asset health",
            asset_id="MTR-001",
        )
    )


# ============================================================
# 3. MAINTENANCE AGENT TESTS
# ============================================================

@pytest.mark.asyncio
async def test_maintenance_agent_success(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.MAINTENANCE,
        assets["MTR-001"],
    )

    assert result.success
    assert result.agent == AgentType.MAINTENANCE


@pytest.mark.asyncio
async def test_maintenance_high_priority(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.MAINTENANCE,
        assets["GEN-001"],
    )

    assert result.data["priority"] == "high"


@pytest.mark.asyncio
async def test_maintenance_low_priority(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.MAINTENANCE,
        assets["PMP-001"],
    )

    assert result.data["priority"] == "low"


# ============================================================
# 4. ANOMALY DETECTION TESTS
# ============================================================

@pytest.mark.asyncio
async def test_anomaly_detected(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.ANOMALY,
        assets["GEN-001"],
    )

    assert result.data["anomaly_detected"] is True


@pytest.mark.asyncio
async def test_anomaly_identifies_temperature(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.ANOMALY,
        assets["GEN-001"],
    )

    assert "temperature" in result.data["anomalous_sensors"]


@pytest.mark.asyncio
async def test_anomaly_identifies_vibration(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.ANOMALY,
        assets["GEN-001"],
    )

    assert "vibration" in result.data["anomalous_sensors"]


@pytest.mark.asyncio
async def test_normal_asset_has_no_anomaly(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.ANOMALY,
        assets["PMP-001"],
    )

    assert result.data["anomaly_detected"] is False


# ============================================================
# 5. RUL AGENT TESTS
# ============================================================

@pytest.mark.asyncio
async def test_rul_prediction_returned(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.RUL,
        assets["MTR-001"],
    )

    assert result.data["rul_months"] == 48


@pytest.mark.asyncio
async def test_rul_is_nonnegative(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.RUL,
        assets["GEN-001"],
    )

    assert result.data["rul_months"] >= 0


@pytest.mark.asyncio
async def test_failure_probability_valid(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.RUL,
        assets["GEN-001"],
    )

    probability = result.data["failure_probability"]
    assert 0 <= probability <= 1


# ============================================================
# 6. SAFETY AGENT TESTS
# ============================================================

@pytest.mark.asyncio
async def test_safety_detects_hazard(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.SAFETY,
        assets["GEN-001"],
    )

    assert result.data["hazard_detected"] is True


@pytest.mark.asyncio
async def test_safety_normal_equipment(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.SAFETY,
        assets["PMP-001"],
    )

    assert result.data["hazard_detected"] is False


@pytest.mark.asyncio
async def test_safety_severity(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.SAFETY,
        assets["GEN-001"],
    )

    assert result.data["severity"] == "high"


# ============================================================
# 7. EMERGENCY AGENT TESTS
# ============================================================

@pytest.mark.asyncio
async def test_emergency_no_false_shutdown(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.EMERGENCY,
        assets["GEN-001"],
    )

    assert result.data["automatic_shutdown"] is False


@pytest.mark.asyncio
async def test_emergency_normal_condition(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.EMERGENCY,
        assets["PMP-001"],
    )

    assert result.data["emergency_detected"] is False


@pytest.mark.asyncio
async def test_emergency_extreme_temperature(pipeline, assets):
    extreme = Asset(
        **{
            **vars(assets["GEN-001"]),
            "temperature": 110,
        }
    )

    result = await pipeline.execute_agent(
        AgentType.EMERGENCY,
        extreme,
    )

    assert result.data["emergency_detected"] is True
    assert result.data["escalation_required"] is True


# ============================================================
# 8. COST INTELLIGENCE TESTS
# ============================================================

@pytest.mark.asyncio
async def test_cost_agent_success(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.COST,
        assets["MTR-001"],
    )

    assert result.success


@pytest.mark.asyncio
async def test_expected_failure_cost_calculation(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.COST,
        assets["MTR-001"],
    )

    expected = 0.32 * 250000

    assert result.data["expected_failure_cost"] == pytest.approx(
        expected
    )


@pytest.mark.asyncio
async def test_cost_currency(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.COST,
        assets["MTR-001"],
    )

    assert result.data["currency"] == "INR"


# ============================================================
# 9. INDUSTRIAL RAG TESTS
# ============================================================

@pytest.mark.asyncio
async def test_rag_agent_success(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.RAG,
        assets["MTR-001"],
    )

    assert result.success


@pytest.mark.asyncio
async def test_rag_returns_citations(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.RAG,
        assets["GEN-001"],
    )

    assert len(result.citations) > 0


@pytest.mark.asyncio
async def test_rag_citations_are_valid(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.RAG,
        assets["GEN-001"],
    )

    assert all(
        citation.startswith("manual://")
        for citation in result.citations
    )


@pytest.mark.asyncio
async def test_rag_answer_not_empty(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.RAG,
        assets["MTR-001"],
    )

    assert result.data["answer"].strip()


# ============================================================
# 10. MULTI-AGENT PIPELINE TESTS
# ============================================================

@pytest.mark.asyncio
async def test_pipeline_success(pipeline):
    result = await pipeline.run(
        AgentRequest(
            query="Check maintenance and safety",
            asset_id="MTR-001",
        )
    )

    assert result.success


@pytest.mark.asyncio
async def test_pipeline_multiple_agent_outputs(pipeline):
    result = await pipeline.run(
        AgentRequest(
            query="Analyze anomaly, RUL and cost",
            asset_id="GEN-001",
        )
    )

    assert len(result.results) == 3


@pytest.mark.asyncio
async def test_pipeline_asset_identity(pipeline):
    result = await pipeline.run(
        AgentRequest(
            query="Check maintenance",
            asset_id="MTR-001",
        )
    )

    assert result.asset_id == "MTR-001"


@pytest.mark.asyncio
async def test_pipeline_final_response(pipeline):
    result = await pipeline.run(
        AgentRequest(
            query="Check safety and cost",
            asset_id="GEN-001",
        )
    )

    assert "agent_outputs" in result.final_response
    assert "safety" in result.final_response["agent_outputs"]
    assert "cost" in result.final_response["agent_outputs"]


@pytest.mark.asyncio
async def test_pipeline_preserves_citations(pipeline):
    result = await pipeline.run(
        AgentRequest(
            query="Search equipment manual",
            asset_id="GEN-001",
        )
    )

    assert len(result.final_response["citations"]) > 0


@pytest.mark.asyncio
async def test_pipeline_no_duplicate_execution(pipeline):
    result = await pipeline.run(
        AgentRequest(
            query="maintenance repair inspection",
            asset_id="MTR-001",
        )
    )

    assert len(result.selected_agents) == 1


# ============================================================
# 11. CONFIDENCE & EVALUATION TESTS
# ============================================================

@pytest.mark.asyncio
async def test_agent_confidence_range(pipeline, assets):
    for agent in [
        AgentType.MAINTENANCE,
        AgentType.ANOMALY,
        AgentType.RUL,
        AgentType.SAFETY,
        AgentType.EMERGENCY,
        AgentType.COST,
        AgentType.RAG,
    ]:
        result = await pipeline.execute_agent(
            agent,
            assets["MTR-001"],
        )

        assert 0 <= result.confidence <= 1


@pytest.mark.asyncio
async def test_all_agent_outputs_are_dictionaries(pipeline):
    result = await pipeline.run(
        AgentRequest(
            query="maintenance anomaly RUL safety emergency cost manual",
            asset_id="GEN-001",
        )
    )

    assert all(
        isinstance(output.data, dict)
        for output in result.results
    )


@pytest.mark.asyncio
async def test_pipeline_agent_count_matches_results(pipeline):
    result = await pipeline.run(
        AgentRequest(
            query="maintenance anomaly RUL safety cost manual",
            asset_id="GEN-001",
        )
    )

    assert len(result.selected_agents) == len(result.results)


# ============================================================
# 12. PIPELINE RELIABILITY TESTS
# ============================================================

@pytest.mark.asyncio
async def test_pipeline_repeatability(pipeline):
    request = AgentRequest(
        query="Check maintenance and cost",
        asset_id="MTR-001",
    )

    first = await pipeline.run(request)
    second = await pipeline.run(request)

    assert first.final_response == second.final_response


@pytest.mark.asyncio
async def test_pipeline_parallel_execution(pipeline):
    requests = [
        AgentRequest(
            query="Check maintenance and safety",
            asset_id=asset_id,
        )
        for asset_id in ["MTR-001", "GEN-001", "PMP-001"]
    ]

    results = await asyncio.gather(
        *(pipeline.run(request) for request in requests)
    )

    assert len(results) == 3
    assert all(result.success for result in results)


@pytest.mark.asyncio
async def test_pipeline_no_unexpected_errors(pipeline):
    result = await pipeline.run(
        AgentRequest(
            query="Analyze RUL, anomaly and maintenance",
            asset_id="GEN-001",
        )
    )

    assert result.errors == []


# ============================================================
# 13. SECURITY & GUARDRAIL TESTS
# ============================================================

def test_request_requires_valid_asset(pipeline):
    with pytest.raises(KeyError):
        pipeline.validate_request(
            AgentRequest(
                query="Analyze asset",
                asset_id="../secrets",
            )
        )


def test_request_length_limit(pipeline):
    with pytest.raises(ValueError):
        pipeline.validate_request(
            AgentRequest(
                query="A" * 10000,
                asset_id="MTR-001",
            )
        )


@pytest.mark.asyncio
async def test_emergency_agent_never_controls_equipment(pipeline, assets):
    result = await pipeline.execute_agent(
        AgentType.EMERGENCY,
        assets["GEN-001"],
    )

    assert result.data["automatic_shutdown"] is False


@pytest.mark.asyncio
async def test_pipeline_returns_structured_output(pipeline):
    result = await pipeline.run(
        AgentRequest(
            query="Check safety",
            asset_id="GEN-001",
        )
    )

    assert isinstance(result.final_response, dict)
    assert isinstance(result.results, list)


# ============================================================
# 14. END-TO-END INDUSTRIAL SCENARIO
# ============================================================

@pytest.mark.asyncio
async def test_end_to_end_industrial_pipeline(pipeline):
    request = AgentRequest(
        query=(
            "Analyze maintenance, sensor anomaly, "
            "remaining useful life, safety, emergency, "
            "replacement cost and equipment manual"
        ),
        asset_id="GEN-001",
        tenant_id="tenant-a",
    )

    result = await pipeline.run(request)

    assert result.success
    assert result.asset_id == "GEN-001"

    expected_agents = {
        AgentType.MAINTENANCE,
        AgentType.ANOMALY,
        AgentType.RUL,
        AgentType.SAFETY,
        AgentType.EMERGENCY,
        AgentType.COST,
        AgentType.RAG,
    }

    assert set(result.selected_agents) == expected_agents

    outputs = result.final_response["agent_outputs"]

    assert outputs["maintenance"]["priority"] == "high"
    assert outputs["anomaly"]["anomaly_detected"] is True
    assert outputs["rul"]["rul_months"] == 24
    assert outputs["safety"]["hazard_detected"] is True

    assert outputs["emergency"]["automatic_shutdown"] is False
    assert outputs["cost"]["expected_failure_cost"] > 0

    assert len(result.final_response["citations"]) > 0
    assert result.errors == []
