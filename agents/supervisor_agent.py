# app/agents/supervisor_agent.py

from __future__ import annotations

import asyncio
import inspect
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence
from uuid import UUID, uuid4


# =========================================================
# Utilities
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def probability(value: float, *, name: str) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")

    return value


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if is_dataclass(value):
        return json_safe(asdict(value))

    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump(mode="json"))
        except Exception:
            return json_safe(value.model_dump())

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def get_value(
    value: Any,
    key: str,
    default: Any = None,
) -> Any:
    if value is None:
        return default

    if isinstance(value, Mapping):
        return value.get(key, default)

    return getattr(value, key, default)


# =========================================================
# Enums
# =========================================================

class AgentName(str, Enum):
    ASSET_HEALTH = "asset_health_agent"
    FAILURE_ANALYSIS = "failure_analysis_agent"
    INSPECTION = "inspection_agent"
    SAFETY = "safety_agent"
    INCIDENT = "incident_agent"
    EMERGENCY_RESPONSE = "emergency_response_agent"
    MAINTENANCE = "maintenance_agent"
    REPAIR_REPLACEMENT = "repair_replacement_agent"
    COST = "cost_agent"
    REPORT = "report_agent"


class WorkflowType(str, Enum):
    AUTO = "auto"

    ASSET_HEALTH = "asset_health"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    INSPECTION = "inspection"

    SAFETY = "safety"
    INCIDENT = "incident"
    EMERGENCY = "emergency"

    REPAIR_REPLACEMENT = "repair_replacement"
    COST_OPTIMIZATION = "cost_optimization"

    FULL_ASSET_ASSESSMENT = "full_asset_assessment"


class SupervisorStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_FOR_REVIEW = "waiting_for_review"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


class SupervisorSeverity(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SupervisorDecision(str, Enum):
    CONTINUE_OPERATION = "continue_operation"
    MONITOR = "monitor"
    INSPECT = "inspect"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    REPLACE = "replace"
    RESTRICT_OPERATION = "restrict_operation"
    REQUEST_SHUTDOWN_REVIEW = "request_shutdown_review"
    EMERGENCY_RESPONSE = "emergency_response"
    HUMAN_REVIEW = "human_review"


class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    DEPENDENCY_AWARE = "dependency_aware"


class ToolRiskLevel(str, Enum):
    READ_ONLY = "read_only"
    BUSINESS_ACTION = "business_action"
    SAFETY_RELATED = "safety_related"
    CRITICAL_CONTROL = "critical_control"


# =========================================================
# Agent contract
# =========================================================

class AgentProtocol(Protocol):

    async def run(self, data: Any) -> Any:
        ...


AgentInputBuilder = Callable[
    ["SupervisorContext", "SupervisorTask"],
    Any | Awaitable[Any],
]


# =========================================================
# Input
# =========================================================

@dataclass(slots=True)
class SupervisorInput:
    workflow: WorkflowType = WorkflowType.AUTO

    asset_id: str | None = None
    asset_name: str | None = None

    site_id: str | None = None
    zone_id: str | None = None

    telemetry: Mapping[str, Any] | None = None
    sensor_data: Mapping[str, Any] | None = None

    image_analysis: Mapping[str, Any] | None = None
    inspection_data: Mapping[str, Any] | None = None

    anomaly_data: Mapping[str, Any] | None = None
    failure_prediction: Mapping[str, Any] | None = None
    rul_prediction: Mapping[str, Any] | None = None

    safety_data: Mapping[str, Any] | None = None
    incident_data: Mapping[str, Any] | None = None

    maintenance_history: Sequence[Mapping[str, Any]] = field(
        default_factory=list
    )

    asset_context: Mapping[str, Any] | None = None
    cost_context: Mapping[str, Any] | None = None

    requested_agents: list[AgentName] = field(default_factory=list)

    skip_agents: list[AgentName] = field(default_factory=list)

    force_human_review: bool = False

    allow_business_actions: bool = False

    # Deliberately false by default.
    allow_safety_control_requests: bool = False

    request_id: str | None = None
    trace_id: str | None = None

    user_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


# =========================================================
# Task
# =========================================================

@dataclass(slots=True)
class SupervisorTask:
    task_id: str

    agent: AgentName

    dependencies: list[AgentName] = field(default_factory=list)

    required: bool = False

    timeout_seconds: float = 30.0

    max_attempts: int = 1

    risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY

    status: TaskStatus = TaskStatus.PENDING

    attempts: int = 0

    started_at: datetime | None = None
    completed_at: datetime | None = None

    error: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


# =========================================================
# Execution
# =========================================================

@dataclass(slots=True)
class AgentExecution:
    execution_id: str

    task_id: str
    agent: AgentName

    status: TaskStatus

    attempts: int

    duration_ms: float

    output: Any = None

    error: str | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


# =========================================================
# Context
# =========================================================

@dataclass(slots=True)
class SupervisorContext:
    workflow_id: str

    request: SupervisorInput

    outputs: dict[AgentName, Any] = field(default_factory=dict)

    executions: dict[AgentName, AgentExecution] = field(
        default_factory=dict
    )

    shared: dict[str, Any] = field(default_factory=dict)

    started_at: datetime = field(default_factory=utc_now)

    def output(
        self,
        agent: AgentName,
        default: Any = None,
    ) -> Any:
        return self.outputs.get(agent, default)

    def has_success(self, agent: AgentName) -> bool:
        execution = self.executions.get(agent)

        return bool(
            execution
            and execution.status == TaskStatus.SUCCESS
        )


# =========================================================
# Decision result
# =========================================================

@dataclass(slots=True)
class SupervisorDecisionResult:
    decision: SupervisorDecision

    severity: SupervisorSeverity

    confidence: float

    reason: str

    asset_id: str | None

    human_review_required: bool

    safety_review_required: bool

    emergency_detected: bool

    recommended_next_agents: list[AgentName] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(default_factory=dict)


# =========================================================
# Supervisor result
# =========================================================

@dataclass(slots=True)
class SupervisorResult:
    workflow_id: str

    workflow: WorkflowType

    status: SupervisorStatus

    asset_id: str | None
    site_id: str | None

    decision: SupervisorDecisionResult

    tasks: list[SupervisorTask]

    executions: list[AgentExecution]

    successful_agents: list[AgentName]

    failed_agents: list[AgentName]

    skipped_agents: list[AgentName]

    final_report: Any | None

    started_at: datetime
    completed_at: datetime

    duration_ms: float

    request_id: str | None = None
    trace_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# External services
# =========================================================

class SupervisorRepository(Protocol):

    async def save_workflow(
        self,
        result: SupervisorResult,
    ) -> Any:
        ...


class AuditService(Protocol):

    async def record(
        self,
        *,
        event_type: str,
        actor_id: str | None,
        resource_id: str,
        data: Mapping[str, Any],
    ) -> Any:
        ...


class EventPublisher(Protocol):

    async def publish(
        self,
        *,
        topic: str,
        event: Mapping[str, Any],
    ) -> Any:
        ...


# =========================================================
# Registered agent
# =========================================================

@dataclass(slots=True)
class RegisteredAgent:
    name: AgentName
    agent: AgentProtocol

    input_builder: AgentInputBuilder | None = None

    enabled: bool = True

    risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY

    default_timeout_seconds: float = 30.0


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class SupervisorConfig:
    execution_mode: ExecutionMode = (
        ExecutionMode.DEPENDENCY_AWARE
    )

    default_timeout_seconds: float = 30.0

    failure_analysis_timeout_seconds: float = 45.0
    inspection_timeout_seconds: float = 60.0
    report_timeout_seconds: float = 60.0

    default_max_attempts: int = 2

    stop_on_emergency: bool = False

    generate_final_report: bool = True

    require_human_review_for_replacement: bool = True
    require_human_review_for_shutdown: bool = True

    minimum_decision_confidence: float = 0.60

    critical_failure_probability: float = 0.90
    high_failure_probability: float = 0.70

    critical_rul_hours: float = 24.0
    warning_rul_hours: float = 168.0

    critical_health_score: float = 30.0
    warning_health_score: float = 60.0

    persist_workflow: bool = True
    publish_events: bool = True
    audit_workflow: bool = True


# =========================================================
# Supervisor
# =========================================================

class SupervisorAgent:

    def __init__(
        self,
        *,
        config: SupervisorConfig | None = None,
        repository: SupervisorRepository | None = None,
        audit_service: AuditService | None = None,
        event_publisher: EventPublisher | None = None,
    ) -> None:

        self.config = config or SupervisorConfig()

        self.repository = repository
        self.audit_service = audit_service
        self.event_publisher = event_publisher

        self._agents: dict[
            AgentName,
            RegisteredAgent,
        ] = {}

    # =====================================================
    # Registration
    # =====================================================

    def register(
        self,
        name: AgentName,
        agent: AgentProtocol,
        *,
        input_builder: AgentInputBuilder | None = None,
        enabled: bool = True,
        risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY,
        timeout_seconds: float | None = None,
    ) -> None:

        self._agents[name] = RegisteredAgent(
            name=name,
            agent=agent,
            input_builder=input_builder,
            enabled=enabled,
            risk_level=risk_level,
            default_timeout_seconds=(
                timeout_seconds
                or self.config.default_timeout_seconds
            ),
        )

    def unregister(self, name: AgentName) -> None:
        self._agents.pop(name, None)

    def registered_agents(self) -> list[AgentName]:
        return list(self._agents)

    # =====================================================
    # Main workflow
    # =====================================================

    async def run(
        self,
        data: SupervisorInput,
    ) -> SupervisorResult:

        started = utc_now()

        workflow_id = new_id()

        context = SupervisorContext(
            workflow_id=workflow_id,
            request=data,
            started_at=started,
        )

        tasks = self._build_plan(data)

        await self._audit(
            event_type="supervisor.workflow.started",
            context=context,
            data={
                "workflow": data.workflow.value,
                "task_count": len(tasks),
            },
        )

        await self._publish(
            "industrial.supervisor.started",
            {
                "workflow_id": workflow_id,
                "workflow": data.workflow.value,
                "asset_id": data.asset_id,
                "site_id": data.site_id,
            },
        )

        await self._execute_plan(
            context=context,
            tasks=tasks,
        )

        decision = self._make_decision(context)

        final_report = None

        if (
            self.config.generate_final_report
            and AgentName.REPORT in self._agents
            and AgentName.REPORT not in data.skip_agents
        ):
            final_report = await self._run_final_report(
                context=context,
                tasks=tasks,
                decision=decision,
            )

        completed = utc_now()

        successful = [
            execution.agent
            for execution in context.executions.values()
            if execution.status == TaskStatus.SUCCESS
        ]

        failed = [
            execution.agent
            for execution in context.executions.values()
            if execution.status
            in {
                TaskStatus.FAILED,
                TaskStatus.TIMEOUT,
            }
        ]

        skipped = [
            task.agent
            for task in tasks
            if task.status
            in {
                TaskStatus.SKIPPED,
                TaskStatus.BLOCKED,
            }
        ]

        status = self._workflow_status(
            tasks=tasks,
            decision=decision,
        )

        if (
            decision.human_review_required
            and status
            == SupervisorStatus.COMPLETED
        ):
            status = SupervisorStatus.WAITING_FOR_REVIEW

        result = SupervisorResult(
            workflow_id=workflow_id,

            workflow=data.workflow,

            status=status,

            asset_id=data.asset_id,
            site_id=data.site_id,

            decision=decision,

            tasks=tasks,

            executions=list(
                context.executions.values()
            ),

            successful_agents=successful,
            failed_agents=failed,
            skipped_agents=skipped,

            final_report=final_report,

            started_at=started,
            completed_at=completed,

            duration_ms=(
                completed - started
            ).total_seconds() * 1000.0,

            request_id=data.request_id,
            trace_id=data.trace_id,

            metadata={
                **data.metadata,
                "registered_agents": [
                    item.value
                    for item in self._agents
                ],
            },
        )

        await self._post_process(result, data)

        return result

    # =====================================================
    # Planning
    # =====================================================

    def _build_plan(
        self,
        data: SupervisorInput,
    ) -> list[SupervisorTask]:

        if data.requested_agents:
            selected = list(
                dict.fromkeys(data.requested_agents)
            )
        else:
            selected = self._workflow_agents(
                data.workflow,
                data,
            )

        selected = [
            agent
            for agent in selected
            if agent not in data.skip_agents
        ]

        # Report is generated after final decision.
        selected = [
            agent
            for agent in selected
            if agent != AgentName.REPORT
        ]

        dependencies = self._dependencies()

        tasks: list[SupervisorTask] = []

        selected_set = set(selected)

        for agent in selected:
            registered = self._agents.get(agent)

            risk = (
                registered.risk_level
                if registered
                else self._default_risk(agent)
            )

            timeout = self._timeout_for(agent)

            deps = [
                dependency
                for dependency
                in dependencies.get(agent, [])
                if dependency in selected_set
            ]

            tasks.append(
                SupervisorTask(
                    task_id=new_id(),
                    agent=agent,
                    dependencies=deps,
                    required=self._required_agent(
                        agent,
                        data.workflow,
                    ),
                    timeout_seconds=timeout,
                    max_attempts=(
                        self.config.default_max_attempts
                    ),
                    risk_level=risk,
                )
            )

        return tasks

    def _workflow_agents(
        self,
        workflow: WorkflowType,
        data: SupervisorInput,
    ) -> list[AgentName]:

        if workflow == WorkflowType.ASSET_HEALTH:
            return [
                AgentName.ASSET_HEALTH,
            ]

        if workflow == WorkflowType.INSPECTION:
            return [
                AgentName.INSPECTION,
                AgentName.ASSET_HEALTH,
                AgentName.MAINTENANCE,
                AgentName.REPAIR_REPLACEMENT,
                AgentName.COST,
            ]

        if workflow == WorkflowType.SAFETY:
            return [
                AgentName.SAFETY,
                AgentName.INCIDENT,
            ]

        if workflow == WorkflowType.INCIDENT:
            return [
                AgentName.SAFETY,
                AgentName.INCIDENT,
                AgentName.EMERGENCY_RESPONSE,
            ]

        if workflow == WorkflowType.EMERGENCY:
            return [
                AgentName.SAFETY,
                AgentName.INCIDENT,
                AgentName.EMERGENCY_RESPONSE,
            ]

        if workflow == WorkflowType.REPAIR_REPLACEMENT:
            return [
                AgentName.ASSET_HEALTH,
                AgentName.FAILURE_ANALYSIS,
                AgentName.INSPECTION,
                AgentName.MAINTENANCE,
                AgentName.REPAIR_REPLACEMENT,
                AgentName.COST,
            ]

        if workflow == WorkflowType.COST_OPTIMIZATION:
            return [
                AgentName.ASSET_HEALTH,
                AgentName.FAILURE_ANALYSIS,
                AgentName.MAINTENANCE,
                AgentName.REPAIR_REPLACEMENT,
                AgentName.COST,
            ]

        if workflow == WorkflowType.PREDICTIVE_MAINTENANCE:
            return [
                AgentName.ASSET_HEALTH,
                AgentName.FAILURE_ANALYSIS,
                AgentName.INSPECTION,
                AgentName.MAINTENANCE,
                AgentName.REPAIR_REPLACEMENT,
                AgentName.COST,
            ]

        if workflow == WorkflowType.FULL_ASSET_ASSESSMENT:
            return [
                AgentName.ASSET_HEALTH,
                AgentName.FAILURE_ANALYSIS,
                AgentName.INSPECTION,
                AgentName.SAFETY,
                AgentName.INCIDENT,
                AgentName.EMERGENCY_RESPONSE,
                AgentName.MAINTENANCE,
                AgentName.REPAIR_REPLACEMENT,
                AgentName.COST,
            ]

        # AUTO
        agents = [
            AgentName.ASSET_HEALTH,
            AgentName.FAILURE_ANALYSIS,
        ]

        if (
            data.image_analysis
            or data.inspection_data
        ):
            agents.append(
                AgentName.INSPECTION
            )

        if data.safety_data:
            agents.append(
                AgentName.SAFETY
            )

        if data.incident_data:
            agents.extend([
                AgentName.INCIDENT,
                AgentName.EMERGENCY_RESPONSE,
            ])

        agents.extend([
            AgentName.MAINTENANCE,
            AgentName.REPAIR_REPLACEMENT,
            AgentName.COST,
        ])

        return list(dict.fromkeys(agents))

    @staticmethod
    def _dependencies() -> dict[
        AgentName,
        list[AgentName],
    ]:
        return {
            AgentName.FAILURE_ANALYSIS: [
                AgentName.ASSET_HEALTH,
            ],

            AgentName.MAINTENANCE: [
                AgentName.ASSET_HEALTH,
                AgentName.FAILURE_ANALYSIS,
            ],

            AgentName.REPAIR_REPLACEMENT: [
                AgentName.ASSET_HEALTH,
                AgentName.FAILURE_ANALYSIS,
                AgentName.MAINTENANCE,
            ],

            AgentName.COST: [
                AgentName.REPAIR_REPLACEMENT,
            ],

            AgentName.INCIDENT: [
                AgentName.SAFETY,
            ],

            AgentName.EMERGENCY_RESPONSE: [
                AgentName.SAFETY,
                AgentName.INCIDENT,
            ],
        }

    def _required_agent(
        self,
        agent: AgentName,
        workflow: WorkflowType,
    ) -> bool:

        if workflow == WorkflowType.EMERGENCY:
            return agent in {
                AgentName.SAFETY,
                AgentName.EMERGENCY_RESPONSE,
            }

        if workflow == WorkflowType.SAFETY:
            return agent == AgentName.SAFETY

        if workflow == WorkflowType.ASSET_HEALTH:
            return agent == AgentName.ASSET_HEALTH

        return agent in {
            AgentName.ASSET_HEALTH,
        }

    def _timeout_for(
        self,
        agent: AgentName,
    ) -> float:

        if agent == AgentName.FAILURE_ANALYSIS:
            return self.config.failure_analysis_timeout_seconds

        if agent == AgentName.INSPECTION:
            return self.config.inspection_timeout_seconds

        if agent == AgentName.REPORT:
            return self.config.report_timeout_seconds

        registered = self._agents.get(agent)

        if registered:
            return registered.default_timeout_seconds

        return self.config.default_timeout_seconds

    @staticmethod
    def _default_risk(
        agent: AgentName,
    ) -> ToolRiskLevel:

        if agent == AgentName.EMERGENCY_RESPONSE:
            return ToolRiskLevel.SAFETY_RELATED

        if agent in {
            AgentName.MAINTENANCE,
            AgentName.REPAIR_REPLACEMENT,
            AgentName.COST,
            AgentName.INCIDENT,
        }:
            return ToolRiskLevel.BUSINESS_ACTION

        return ToolRiskLevel.READ_ONLY

    # =====================================================
    # Plan execution
    # =====================================================

    async def _execute_plan(
        self,
        *,
        context: SupervisorContext,
        tasks: list[SupervisorTask],
    ) -> None:

        pending = {
            task.agent: task
            for task in tasks
        }

        while pending:
            progressed = False

            for agent, task in list(pending.items()):
                dependency_state = (
                    self._dependency_state(
                        task,
                        context,
                    )
                )

                if dependency_state == "waiting":
                    continue

                if dependency_state == "failed":
                    task.status = TaskStatus.BLOCKED
                    task.error = (
                        "One or more required "
                        "dependencies failed."
                    )

                    context.executions[agent] = (
                        AgentExecution(
                            execution_id=new_id(),
                            task_id=task.task_id,
                            agent=agent,
                            status=TaskStatus.BLOCKED,
                            attempts=0,
                            duration_ms=0.0,
                            error=task.error,
                        )
                    )

                    pending.pop(agent)
                    progressed = True
                    continue

                await self._execute_task(
                    context=context,
                    task=task,
                )

                pending.pop(agent)
                progressed = True

                if (
                    self.config.stop_on_emergency
                    and self._emergency_detected(
                        context
                    )
                ):
                    for remaining in pending.values():
                        remaining.status = (
                            TaskStatus.SKIPPED
                        )

                        remaining.error = (
                            "Workflow stopped after "
                            "emergency detection."
                        )

                    pending.clear()
                    return

            if not progressed:
                # Dependency cycle or impossible plan.
                for task in pending.values():
                    task.status = TaskStatus.BLOCKED
                    task.error = (
                        "Unresolvable dependency "
                        "graph."
                    )

                    context.executions[
                        task.agent
                    ] = AgentExecution(
                        execution_id=new_id(),
                        task_id=task.task_id,
                        agent=task.agent,
                        status=TaskStatus.BLOCKED,
                        attempts=0,
                        duration_ms=0.0,
                        error=task.error,
                    )

                break

    @staticmethod
    def _dependency_state(
        task: SupervisorTask,
        context: SupervisorContext,
    ) -> str:

        if not task.dependencies:
            return "ready"

        for dependency in task.dependencies:
            execution = context.executions.get(
                dependency
            )

            if execution is None:
                return "waiting"

            if execution.status != TaskStatus.SUCCESS:
                return "failed"

        return "ready"

    async def _execute_task(
        self,
        *,
        context: SupervisorContext,
        task: SupervisorTask,
    ) -> None:

        registered = self._agents.get(task.agent)

        if (
            registered is None
            or not registered.enabled
        ):
            task.status = TaskStatus.SKIPPED
            task.error = (
                f"{task.agent.value} is not registered "
                "or is disabled."
            )

            context.executions[task.agent] = (
                AgentExecution(
                    execution_id=new_id(),
                    task_id=task.task_id,
                    agent=task.agent,
                    status=TaskStatus.SKIPPED,
                    attempts=0,
                    duration_ms=0.0,
                    error=task.error,
                )
            )

            return

        if not self._authorized(
            task,
            context.request,
        ):
            task.status = TaskStatus.SKIPPED
            task.error = (
                "Agent execution blocked by "
                "supervisor authorization policy."
            )

            context.executions[task.agent] = (
                AgentExecution(
                    execution_id=new_id(),
                    task_id=task.task_id,
                    agent=task.agent,
                    status=TaskStatus.SKIPPED,
                    attempts=0,
                    duration_ms=0.0,
                    error=task.error,
                )
            )

            return

        task.status = TaskStatus.RUNNING
        task.started_at = utc_now()

        started = task.started_at

        last_error: str | None = None

        for attempt in range(
            1,
            max(task.max_attempts, 1) + 1,
        ):
            task.attempts = attempt

            try:
                payload = await self._build_agent_input(
                    registered,
                    context,
                    task,
                )

                output = await asyncio.wait_for(
                    registered.agent.run(payload),
                    timeout=task.timeout_seconds,
                )

                completed = utc_now()

                task.status = TaskStatus.SUCCESS
                task.completed_at = completed

                context.outputs[task.agent] = output

                context.executions[task.agent] = (
                    AgentExecution(
                        execution_id=new_id(),
                        task_id=task.task_id,
                        agent=task.agent,
                        status=TaskStatus.SUCCESS,
                        attempts=attempt,
                        duration_ms=(
                            completed - started
                        ).total_seconds() * 1000.0,
                        output=output,
                        started_at=started,
                        completed_at=completed,
                    )
                )

                await self._audit(
                    event_type=(
                        "supervisor.agent.completed"
                    ),
                    context=context,
                    data={
                        "agent": task.agent.value,
                        "attempt": attempt,
                    },
                )

                return

            except asyncio.TimeoutError:
                last_error = (
                    f"{task.agent.value} timed out "
                    f"after {task.timeout_seconds}s."
                )

            except Exception as exc:
                last_error = (
                    f"{type(exc).__name__}: {exc}"
                )

            if attempt < task.max_attempts:
                await asyncio.sleep(
                    min(0.25 * (2 ** (attempt - 1)), 2.0)
                )

        completed = utc_now()

        task.status = (
            TaskStatus.TIMEOUT
            if last_error
            and "timed out" in last_error
            else TaskStatus.FAILED
        )

        task.error = last_error
        task.completed_at = completed

        context.executions[task.agent] = (
            AgentExecution(
                execution_id=new_id(),
                task_id=task.task_id,
                agent=task.agent,
                status=task.status,
                attempts=task.attempts,
                duration_ms=(
                    completed - started
                ).total_seconds() * 1000.0,
                error=last_error,
                started_at=started,
                completed_at=completed,
            )
        )

        await self._audit(
            event_type="supervisor.agent.failed",
            context=context,
            data={
                "agent": task.agent.value,
                "error": last_error,
            },
        )

    async def _build_agent_input(
        self,
        registered: RegisteredAgent,
        context: SupervisorContext,
        task: SupervisorTask,
    ) -> Any:

        if registered.input_builder is None:
            return self._default_agent_payload(
                context,
                task,
            )

        value = registered.input_builder(
            context,
            task,
        )

        if inspect.isawaitable(value):
            return await value

        return value

    # =====================================================
    # Default payload
    # =====================================================

    @staticmethod
    def _default_agent_payload(
        context: SupervisorContext,
        task: SupervisorTask,
    ) -> dict[str, Any]:

        request = context.request

        return {
            "workflow_id": context.workflow_id,

            "agent": task.agent.value,

            "asset_id": request.asset_id,
            "asset_name": request.asset_name,

            "site_id": request.site_id,
            "zone_id": request.zone_id,

            "telemetry": request.telemetry,
            "sensor_data": request.sensor_data,

            "image_analysis": request.image_analysis,
            "inspection_data": request.inspection_data,

            "anomaly_data": request.anomaly_data,
            "failure_prediction": request.failure_prediction,
            "rul_prediction": request.rul_prediction,

            "safety_data": request.safety_data,
            "incident_data": request.incident_data,

            "maintenance_history": list(
                request.maintenance_history
            ),

            "asset_context": request.asset_context,
            "cost_context": request.cost_context,

            "previous_outputs": {
                agent.value: json_safe(output)
                for agent, output
                in context.outputs.items()
            },

            "request_id": request.request_id,
            "trace_id": request.trace_id,

            "metadata": {
                **request.metadata,
                "supervisor_workflow_id":
                    context.workflow_id,
            },
        }

    # =====================================================
    # Authorization
    # =====================================================

    @staticmethod
    def _authorized(
        task: SupervisorTask,
        request: SupervisorInput,
    ) -> bool:

        if (
            task.risk_level
            == ToolRiskLevel.READ_ONLY
        ):
            return True

        if (
            task.risk_level
            == ToolRiskLevel.BUSINESS_ACTION
        ):
            # Agent may still calculate/recommend.
            # Actual side effects should remain disabled
            # in the specialized agent unless explicitly
            # authorized.
            return True

        if (
            task.risk_level
            == ToolRiskLevel.SAFETY_RELATED
        ):
            # Safety analysis and emergency planning
            # are allowed. Direct physical control is not.
            return True

        if (
            task.risk_level
            == ToolRiskLevel.CRITICAL_CONTROL
        ):
            return (
                request.allow_safety_control_requests
            )

        return False

    # =====================================================
    # Final decision
    # =====================================================

    def _make_decision(
        self,
        context: SupervisorContext,
    ) -> SupervisorDecisionResult:

        request = context.request

        emergency = self._emergency_detected(
            context
        )

        safety_risk = self._safety_risk(
            context
        )

        failure_probability = (
            self._failure_probability(context)
        )

        rul = self._rul(context)

        health_score = self._health_score(
            context
        )

        repair_decision = (
            self._repair_decision(context)
        )

        confidence_values = (
            self._agent_confidences(context)
        )

        confidence = (
            sum(confidence_values)
            / len(confidence_values)
            if confidence_values
            else 0.50
        )

        # ---------------------------------------------
        # Emergency
        # ---------------------------------------------

        if emergency:
            return SupervisorDecisionResult(
                decision=(
                    SupervisorDecision
                    .EMERGENCY_RESPONSE
                ),

                severity=(
                    SupervisorSeverity.EMERGENCY
                ),

                confidence=confidence,

                reason=(
                    "Emergency or critical safety "
                    "condition detected by the "
                    "specialized safety workflow."
                ),

                asset_id=request.asset_id,

                human_review_required=True,
                safety_review_required=True,
                emergency_detected=True,

                recommended_next_agents=[
                    AgentName.EMERGENCY_RESPONSE,
                    AgentName.INCIDENT,
                    AgentName.REPORT,
                ],
            )

        # ---------------------------------------------
        # Critical safety
        # ---------------------------------------------

        if safety_risk >= 0.90:
            return SupervisorDecisionResult(
                decision=(
                    SupervisorDecision
                    .REQUEST_SHUTDOWN_REVIEW
                ),

                severity=(
                    SupervisorSeverity.CRITICAL
                ),

                confidence=confidence,

                reason=(
                    "Critical safety risk requires "
                    "qualified safety review and "
                    "evaluation of the approved "
                    "safe-state procedure."
                ),

                asset_id=request.asset_id,

                human_review_required=True,
                safety_review_required=True,
                emergency_detected=False,

                recommended_next_agents=[
                    AgentName.SAFETY,
                    AgentName.INCIDENT,
                ],
            )

        # ---------------------------------------------
        # Explicit repair/replacement output
        # ---------------------------------------------

        if repair_decision == "replace":
            return SupervisorDecisionResult(
                decision=SupervisorDecision.REPLACE,

                severity=(
                    SupervisorSeverity.HIGH
                ),

                confidence=confidence,

                reason=(
                    "Repair/replacement analysis "
                    "recommends asset or component "
                    "replacement."
                ),

                asset_id=request.asset_id,

                human_review_required=(
                    self.config
                    .require_human_review_for_replacement
                ),

                safety_review_required=(
                    safety_risk >= 0.60
                ),

                emergency_detected=False,

                recommended_next_agents=[
                    AgentName.COST,
                    AgentName.REPORT,
                ],
            )

        if repair_decision == "repair":
            return SupervisorDecisionResult(
                decision=SupervisorDecision.REPAIR,

                severity=(
                    SupervisorSeverity.HIGH
                ),

                confidence=confidence,

                reason=(
                    "Repair/replacement analysis "
                    "recommends repair."
                ),

                asset_id=request.asset_id,

                human_review_required=(
                    confidence
                    < self.config
                    .minimum_decision_confidence
                ),

                safety_review_required=(
                    safety_risk >= 0.60
                ),

                emergency_detected=False,

                recommended_next_agents=[
                    AgentName.MAINTENANCE,
                    AgentName.COST,
                ],
            )

        # ---------------------------------------------
        # Failure / RUL
        # ---------------------------------------------

        if (
            failure_probability
            >= self.config
            .critical_failure_probability
            or (
                rul is not None
                and rul
                <= self.config
                .critical_rul_hours
            )
        ):
            return SupervisorDecisionResult(
                decision=(
                    SupervisorDecision.MAINTENANCE
                ),

                severity=(
                    SupervisorSeverity.CRITICAL
                ),

                confidence=confidence,

                reason=(
                    "Critical failure probability "
                    "or remaining-useful-life "
                    "threshold has been reached."
                ),

                asset_id=request.asset_id,

                human_review_required=True,

                safety_review_required=(
                    safety_risk >= 0.60
                ),

                emergency_detected=False,

                recommended_next_agents=[
                    AgentName.MAINTENANCE,
                    AgentName.REPAIR_REPLACEMENT,
                ],
            )

        if (
            failure_probability
            >= self.config
            .high_failure_probability
            or (
                rul is not None
                and rul
                <= self.config
                .warning_rul_hours
            )
        ):
            return SupervisorDecisionResult(
                decision=(
                    SupervisorDecision.MAINTENANCE
                ),

                severity=SupervisorSeverity.HIGH,

                confidence=confidence,

                reason=(
                    "Elevated failure risk or low "
                    "remaining useful life requires "
                    "planned maintenance."
                ),

                asset_id=request.asset_id,

                human_review_required=(
                    confidence
                    < self.config
                    .minimum_decision_confidence
                ),

                safety_review_required=False,

                emergency_detected=False,

                recommended_next_agents=[
                    AgentName.MAINTENANCE,
                ],
            )

        # ---------------------------------------------
        # Asset health
        # ---------------------------------------------

        if (
            health_score is not None
            and health_score
            <= self.config
            .critical_health_score
        ):
            return SupervisorDecisionResult(
                decision=SupervisorDecision.INSPECT,

                severity=SupervisorSeverity.HIGH,

                confidence=confidence,

                reason=(
                    "Asset health is below the "
                    "critical health threshold."
                ),

                asset_id=request.asset_id,

                human_review_required=True,
                safety_review_required=False,
                emergency_detected=False,

                recommended_next_agents=[
                    AgentName.INSPECTION,
                    AgentName.FAILURE_ANALYSIS,
                ],
            )

        if (
            health_score is not None
            and health_score
            <= self.config
            .warning_health_score
        ):
            return SupervisorDecisionResult(
                decision=SupervisorDecision.MONITOR,

                severity=(
                    SupervisorSeverity.MODERATE
                ),

                confidence=confidence,

                reason=(
                    "Asset health degradation "
                    "requires increased monitoring."
                ),

                asset_id=request.asset_id,

                human_review_required=False,
                safety_review_required=False,
                emergency_detected=False,

                recommended_next_agents=[
                    AgentName.ASSET_HEALTH,
                ],
            )

        if (
            confidence
            < self.config
            .minimum_decision_confidence
            or request.force_human_review
        ):
            return SupervisorDecisionResult(
                decision=(
                    SupervisorDecision.HUMAN_REVIEW
                ),

                severity=(
                    SupervisorSeverity.MODERATE
                ),

                confidence=confidence,

                reason=(
                    "Available evidence is "
                    "insufficient for a high-"
                    "confidence autonomous "
                    "recommendation."
                ),

                asset_id=request.asset_id,

                human_review_required=True,
                safety_review_required=False,
                emergency_detected=False,

                recommended_next_agents=[
                    AgentName.INSPECTION,
                ],
            )

        return SupervisorDecisionResult(
            decision=(
                SupervisorDecision.CONTINUE_OPERATION
            ),

            severity=SupervisorSeverity.NORMAL,

            confidence=confidence,

            reason=(
                "No critical safety, failure, RUL "
                "or asset-health condition was "
                "identified."
            ),

            asset_id=request.asset_id,

            human_review_required=False,
            safety_review_required=False,
            emergency_detected=False,

            recommended_next_agents=[],
        )

    # =====================================================
    # Signal extraction
    # =====================================================

    @staticmethod
    def _emergency_detected(
        context: SupervisorContext,
    ) -> bool:

        emergency = context.output(
            AgentName.EMERGENCY_RESPONSE
        )

        if emergency is not None:
            if bool(
                get_value(
                    emergency,
                    "emergency_detected",
                    False,
                )
            ):
                return True

            severity = str(
                get_value(
                    emergency,
                    "severity",
                    "",
                )
            ).lower()

            if severity in {
                "critical",
                "catastrophic",
                "emergency",
            }:
                return True

        safety = context.output(
            AgentName.SAFETY
        )

        if safety is not None:
            if bool(
                get_value(
                    safety,
                    "emergency",
                    False,
                )
            ):
                return True

            severity = str(
                get_value(
                    safety,
                    "severity",
                    "",
                )
            ).lower()

            if severity == "emergency":
                return True

        return False

    @staticmethod
    def _safety_risk(
        context: SupervisorContext,
    ) -> float:

        safety = context.output(
            AgentName.SAFETY
        )

        for key in (
            "risk_score",
            "safety_risk_score",
            "overall_risk",
        ):
            value = get_value(
                safety,
                key,
                None,
            )

            if value is not None:
                try:
                    return clamp(float(value))
                except (TypeError, ValueError):
                    pass

        return 0.0

    @staticmethod
    def _failure_probability(
        context: SupervisorContext,
    ) -> float:

        failure = context.output(
            AgentName.FAILURE_ANALYSIS
        )

        for key in (
            "failure_probability",
            "probability",
            "risk_probability",
        ):
            value = get_value(
                failure,
                key,
                None,
            )

            if value is not None:
                try:
                    return clamp(float(value))
                except (TypeError, ValueError):
                    pass

        request_prediction = (
            context.request.failure_prediction
        )

        if request_prediction:
            value = request_prediction.get(
                "failure_probability"
            )

            if value is not None:
                try:
                    return clamp(float(value))
                except (TypeError, ValueError):
                    pass

        return 0.0

    @staticmethod
    def _rul(
        context: SupervisorContext,
    ) -> float | None:

        for agent in (
            AgentName.FAILURE_ANALYSIS,
            AgentName.ASSET_HEALTH,
            AgentName.MAINTENANCE,
        ):
            output = context.output(agent)

            for key in (
                "predicted_rul_hours",
                "rul_hours",
                "remaining_useful_life_hours",
            ):
                value = get_value(
                    output,
                    key,
                    None,
                )

                if value is not None:
                    try:
                        value = float(value)

                        if (
                            math.isfinite(value)
                            and value >= 0
                        ):
                            return value

                    except (TypeError, ValueError):
                        pass

        if context.request.rul_prediction:
            value = (
                context.request.rul_prediction.get(
                    "predicted_rul_hours"
                )
            )

            if value is not None:
                try:
                    value = float(value)

                    if math.isfinite(value):
                        return max(value, 0.0)

                except (TypeError, ValueError):
                    pass

        return None

    @staticmethod
    def _health_score(
        context: SupervisorContext,
    ) -> float | None:

        health = context.output(
            AgentName.ASSET_HEALTH
        )

        value = get_value(
            health,
            "health_score",
            None,
        )

        if value is None:
            return None

        try:
            value = float(value)

            if math.isfinite(value):
                return max(
                    0.0,
                    min(100.0, value),
                )

        except (TypeError, ValueError):
            pass

        return None

    @staticmethod
    def _repair_decision(
        context: SupervisorContext,
    ) -> str | None:

        result = context.output(
            AgentName.REPAIR_REPLACEMENT
        )

        value = get_value(
            result,
            "decision",
            None,
        )

        if isinstance(value, Enum):
            return str(value.value).lower()

        if value is not None:
            return str(value).lower()

        return None

    @staticmethod
    def _agent_confidences(
        context: SupervisorContext,
    ) -> list[float]:

        values: list[float] = []

        for output in context.outputs.values():
            value = get_value(
                output,
                "confidence",
                None,
            )

            if value is None:
                continue

            try:
                values.append(
                    clamp(float(value))
                )
            except (TypeError, ValueError):
                continue

        return values

    # =====================================================
    # Final report
    # =====================================================

    async def _run_final_report(
        self,
        *,
        context: SupervisorContext,
        tasks: list[SupervisorTask],
        decision: SupervisorDecisionResult,
    ) -> Any | None:

        registered = self._agents.get(
            AgentName.REPORT
        )

        if (
            registered is None
            or not registered.enabled
        ):
            return None

        task = SupervisorTask(
            task_id=new_id(),
            agent=AgentName.REPORT,
            dependencies=[],
            required=False,
            timeout_seconds=(
                self.config.report_timeout_seconds
            ),
            max_attempts=1,
            risk_level=ToolRiskLevel.READ_ONLY,
        )

        tasks.append(task)

        context.shared["supervisor_decision"] = (
            decision
        )

        await self._execute_task(
            context=context,
            task=task,
        )

        return context.output(
            AgentName.REPORT
        )

    # =====================================================
    # Workflow status
    # =====================================================

    @staticmethod
    def _workflow_status(
        *,
        tasks: Sequence[SupervisorTask],
        decision: SupervisorDecisionResult,
    ) -> SupervisorStatus:

        required_failed = any(
            task.required
            and task.status
            in {
                TaskStatus.FAILED,
                TaskStatus.TIMEOUT,
                TaskStatus.BLOCKED,
            }
            for task in tasks
        )

        if required_failed:
            return SupervisorStatus.FAILED

        any_failed = any(
            task.status
            in {
                TaskStatus.FAILED,
                TaskStatus.TIMEOUT,
                TaskStatus.BLOCKED,
            }
            for task in tasks
        )

        if any_failed:
            return SupervisorStatus.PARTIAL

        if decision.human_review_required:
            return SupervisorStatus.WAITING_FOR_REVIEW

        return SupervisorStatus.COMPLETED

    # =====================================================
    # Post-processing
    # =====================================================

    async def _post_process(
        self,
        result: SupervisorResult,
        request: SupervisorInput,
    ) -> None:

        self._record_metrics(result)
        self._log_result(result)

        if (
            self.config.persist_workflow
            and self.repository is not None
        ):
            try:
                await self.repository.save_workflow(
                    result
                )
            except Exception:
                pass

        await self._audit_result(
            result,
            request,
        )

        await self._publish_result(result)

    # =====================================================
    # Audit
    # =====================================================

    async def _audit(
        self,
        *,
        event_type: str,
        context: SupervisorContext,
        data: Mapping[str, Any],
    ) -> None:

        if (
            not self.config.audit_workflow
            or self.audit_service is None
        ):
            return

        try:
            await self.audit_service.record(
                event_type=event_type,

                actor_id=(
                    context.request.user_id
                ),

                resource_id=(
                    context.workflow_id
                ),

                data=json_safe(data),
            )

        except Exception:
            pass

    async def _audit_result(
        self,
        result: SupervisorResult,
        request: SupervisorInput,
    ) -> None:

        if (
            not self.config.audit_workflow
            or self.audit_service is None
        ):
            return

        try:
            await self.audit_service.record(
                event_type=(
                    "supervisor.workflow.completed"
                ),

                actor_id=request.user_id,

                resource_id=result.workflow_id,

                data={
                    "status":
                        result.status.value,

                    "decision":
                        result.decision.decision.value,

                    "severity":
                        result.decision.severity.value,

                    "confidence":
                        result.decision.confidence,

                    "successful_agents": [
                        item.value
                        for item
                        in result.successful_agents
                    ],

                    "failed_agents": [
                        item.value
                        for item
                        in result.failed_agents
                    ],
                },
            )

        except Exception:
            pass

    # =====================================================
    # Events
    # =====================================================

    async def _publish(
        self,
        topic: str,
        event: Mapping[str, Any],
    ) -> None:

        if (
            not self.config.publish_events
            or self.event_publisher is None
        ):
            return

        try:
            await self.event_publisher.publish(
                topic=topic,
                event=json_safe(event),
            )

        except Exception:
            pass

    async def _publish_result(
        self,
        result: SupervisorResult,
    ) -> None:

        await self._publish(
            "industrial.supervisor.completed",
            {
                "workflow_id":
                    result.workflow_id,

                "workflow":
                    result.workflow.value,

                "status":
                    result.status.value,

                "asset_id":
                    result.asset_id,

                "site_id":
                    result.site_id,

                "decision":
                    result.decision.decision.value,

                "severity":
                    result.decision.severity.value,

                "confidence":
                    result.decision.confidence,

                "human_review_required":
                    result.decision
                    .human_review_required,

                "safety_review_required":
                    result.decision
                    .safety_review_required,

                "emergency_detected":
                    result.decision
                    .emergency_detected,
            },
        )

    # =====================================================
    # Metrics
    # =====================================================

    @staticmethod
    def _record_metrics(
        result: SupervisorResult,
    ) -> None:

        try:
            from app.observability.metrics import (
                registry,
            )

            registry.counter(
                "supervisor_workflows_total"
            ).inc(
                labels={
                    "workflow":
                        result.workflow.value,

                    "status":
                        result.status.value,

                    "decision":
                        result.decision
                        .decision.value,
                }
            )

            registry.histogram(
                "supervisor_workflow_duration_ms"
            ).observe(
                result.duration_ms,

                labels={
                    "workflow":
                        result.workflow.value,
                },
            )

            registry.histogram(
                "supervisor_decision_confidence"
            ).observe(
                result.decision.confidence,

                labels={
                    "decision":
                        result.decision
                        .decision.value,
                },
            )

            if (
                result.decision
                .emergency_detected
            ):
                registry.counter(
                    "supervisor_emergencies_total"
                ).inc()

        except Exception:
            pass

    # =====================================================
    # Logging
    # =====================================================

    @staticmethod
    def _log_result(
        result: SupervisorResult,
    ) -> None:

        try:
            from app.observability.logging import (
                get_logger,
            )

            logger = get_logger(
                "industrial.supervisor_agent"
            )

            payload = {
                "workflow_id":
                    result.workflow_id,

                "workflow":
                    result.workflow.value,

                "status":
                    result.status.value,

                "asset_id":
                    result.asset_id,

                "decision":
                    result.decision
                    .decision.value,

                "severity":
                    result.decision
                    .severity.value,

                "confidence":
                    result.decision.confidence,

                "human_review_required":
                    result.decision
                    .human_review_required,

                "safety_review_required":
                    result.decision
                    .safety_review_required,

                "successful_agents": [
                    item.value
                    for item
                    in result.successful_agents
                ],

                "failed_agents": [
                    item.value
                    for item
                    in result.failed_agents
                ],

                "duration_ms":
                    result.duration_ms,
            }

            if (
                result.decision.severity
                == SupervisorSeverity.EMERGENCY
            ):
                logger.critical(
                    "Supervisor workflow detected "
                    "an emergency condition.",
                    extra={
                        "supervisor": payload
                    },
                )

            elif (
                result.decision.severity
                in {
                    SupervisorSeverity.CRITICAL,
                    SupervisorSeverity.HIGH,
                }
            ):
                logger.warning(
                    "Supervisor workflow completed "
                    "with elevated risk.",
                    extra={
                        "supervisor": payload
                    },
                )

            else:
                logger.info(
                    "Supervisor workflow completed.",
                    extra={
                        "supervisor": payload
                    },
                )

        except Exception:
            pass


# =========================================================
# Factory
# =========================================================

def create_supervisor_agent(
    *,
    config: SupervisorConfig | None = None,
    repository: SupervisorRepository | None = None,
    audit_service: AuditService | None = None,
    event_publisher: EventPublisher | None = None,
) -> SupervisorAgent:

    return SupervisorAgent(
        config=config,
        repository=repository,
        audit_service=audit_service,
        event_publisher=event_publisher,
    )


supervisor_agent = SupervisorAgent()


__all__ = [
    "AgentName",
    "WorkflowType",
    "SupervisorStatus",
    "TaskStatus",
    "SupervisorSeverity",
    "SupervisorDecision",
    "ExecutionMode",
    "ToolRiskLevel",

    "SupervisorInput",
    "SupervisorTask",
    "AgentExecution",
    "SupervisorContext",

    "SupervisorDecisionResult",
    "SupervisorResult",

    "SupervisorRepository",
    "AuditService",
    "EventPublisher",

    "RegisteredAgent",
    "SupervisorConfig",
    "SupervisorAgent",

    "create_supervisor_agent",
    "supervisor_agent",
]