"""
Advanced Multi-Agent Orchestration System

Coordinates multiple specialized AI agents with:
- Parallel execution
- Dynamic task allocation
- Intelligent routing
- Result aggregation
- Conflict resolution
- Performance optimization
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict
import json


class AgentStatus(str, Enum):
    """Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TaskPriority(str, Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExecutionMode(str, Enum):
    """Agent execution modes"""
    PARALLEL = "parallel"  # Execute agents simultaneously
    SEQUENTIAL = "sequential"  # Execute one after another
    CONDITIONAL = "conditional"  # Execute based on conditions
    HIERARCHICAL = "hierarchical"  # Supervisor delegates to workers


@dataclass
class AgentCapability:
    """Agent capabilities and constraints"""
    agent_name: str
    capabilities: List[str]
    max_concurrent_tasks: int = 1
    average_execution_time: float = 5.0  # seconds
    success_rate: float = 0.95
    cost_per_execution: float = 0.0
    required_resources: List[str] = field(default_factory=list)


@dataclass
class Task:
    """Task to be executed by an agent"""
    task_id: str
    task_type: str
    description: str
    priority: TaskPriority
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    status: AgentStatus = AgentStatus.IDLE
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3


@dataclass
class AgentExecution:
    """Agent execution record"""
    execution_id: str
    agent_name: str
    task: Task
    status: AgentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)


@dataclass
class OrchestrationPlan:
    """Execution plan for multiple agents"""
    plan_id: str
    request_id: str
    execution_mode: ExecutionMode
    tasks: List[Task]
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    execution_order: List[List[str]] = field(default_factory=list)  # Stages
    estimated_duration: float = 0.0
    total_cost: float = 0.0


@dataclass
class OrchestrationResult:
    """Complete orchestration result"""
    request_id: str
    plan: OrchestrationPlan
    executions: List[AgentExecution]
    aggregated_results: Dict[str, Any] = field(default_factory=dict)
    status: AgentStatus = AgentStatus.COMPLETED
    total_duration: float = 0.0
    success_rate: float = 1.0
    errors: List[str] = field(default_factory=list)


class MultiAgentOrchestrator:
    """
    Advanced Multi-Agent Orchestration Engine
    
    Features:
    - Parallel agent execution
    - Dynamic task allocation based on agent capabilities
    - Dependency resolution
    - Result aggregation and conflict resolution
    - Performance monitoring and optimization
    - Resource management
    - Fault tolerance and retry logic
    """
    
    def __init__(self):
        # Agent registry
        self.agents: Dict[str, AgentCapability] = {}
        self.agent_instances: Dict[str, Any] = {}
        
        # Execution tracking
        self.active_executions: Dict[str, AgentExecution] = {}
        self.execution_history: List[AgentExecution] = []
        
        # Performance metrics
        self.agent_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Configuration
        self.max_parallel_agents = 10
        self.default_timeout = 60.0  # seconds
        self.enable_caching = True
        self.cache: Dict[str, Any] = {}
    
    def register_agent(
        self,
        agent_name: str,
        agent_instance: Any,
        capabilities: List[str],
        max_concurrent: int = 1,
        avg_execution_time: float = 5.0
    ):
        """Register an agent with the orchestrator"""
        capability = AgentCapability(
            agent_name=agent_name,
            capabilities=capabilities,
            max_concurrent_tasks=max_concurrent,
            average_execution_time=avg_execution_time
        )
        
        self.agents[agent_name] = capability
        self.agent_instances[agent_name] = agent_instance
        
        print(f"✅ Registered agent: {agent_name} with capabilities: {capabilities}")
    
    async def execute_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        mode: ExecutionMode = ExecutionMode.PARALLEL,
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> OrchestrationResult:
        """
        Execute a multi-agent request
        
        Args:
            request: Natural language request or structured task
            context: Additional context for agents
            mode: Execution mode (parallel, sequential, etc.)
            priority: Request priority
            
        Returns:
            Orchestration result with aggregated outputs
        """
        request_id = f"REQ-{uuid.uuid4().hex[:8]}"
        start_time = datetime.utcnow()
        
        print(f"\n🚀 Starting Multi-Agent Orchestration: {request_id}")
        print(f"   Request: {request}")
        print(f"   Mode: {mode}")
        
        # Step 1: Analyze request and decompose into tasks
        tasks = await self._decompose_request(request, context, priority)
        print(f"   Decomposed into {len(tasks)} tasks")
        
        # Step 2: Create orchestration plan
        plan = await self._create_orchestration_plan(
            request_id,
            tasks,
            mode
        )
        print(f"   Execution plan created: {len(plan.execution_order)} stages")
        
        # Step 3: Execute plan
        executions = await self._execute_plan(plan)
        print(f"   Executed {len(executions)} agent calls")
        
        # Step 4: Aggregate results
        aggregated = await self._aggregate_results(executions, request)
        
        # Step 5: Resolve conflicts if any
        final_results = await self._resolve_conflicts(aggregated)
        
        # Calculate metrics
        total_duration = (datetime.utcnow() - start_time).total_seconds()
        success_count = sum(1 for e in executions if e.status == AgentStatus.COMPLETED)
        success_rate = success_count / len(executions) if executions else 0.0
        
        errors = [e.error for e in executions if e.error]
        
        result = OrchestrationResult(
            request_id=request_id,
            plan=plan,
            executions=executions,
            aggregated_results=final_results,
            status=AgentStatus.COMPLETED if success_rate > 0.5 else AgentStatus.FAILED,
            total_duration=total_duration,
            success_rate=success_rate,
            errors=errors
        )
        
        print(f"✅ Orchestration Complete: {request_id}")
        print(f"   Duration: {total_duration:.2f}s")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Results: {len(final_results)} outputs\n")
        
        return result
    
    async def _decompose_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]],
        priority: TaskPriority
    ) -> List[Task]:
        """Decompose request into executable tasks"""
        tasks = []
        
        # Analyze request and identify required agents
        # This is a simplified version - in production, use NLP/LLM
        
        request_lower = request.lower()
        
        # Asset-related tasks
        if "asset" in request_lower or "equipment" in request_lower:
            tasks.append(Task(
                task_id=f"TASK-{uuid.uuid4().hex[:8]}",
                task_type="asset_analysis",
                description="Analyze asset health and status",
                priority=priority,
                inputs={"request": request, "context": context}
            ))
        
        # Maintenance-related tasks
        if "maintenance" in request_lower or "repair" in request_lower:
            tasks.append(Task(
                task_id=f"TASK-{uuid.uuid4().hex[:8]}",
                task_type="maintenance_planning",
                description="Plan maintenance activities",
                priority=priority,
                inputs={"request": request, "context": context}
            ))
        
        # Safety-related tasks
        if "safety" in request_lower or "incident" in request_lower:
            tasks.append(Task(
                task_id=f"TASK-{uuid.uuid4().hex[:8]}",
                task_type="safety_assessment",
                description="Assess safety conditions and risks",
                priority=priority,
                inputs={"request": request, "context": context}
            ))
        
        # Cost-related tasks
        if "cost" in request_lower or "budget" in request_lower or "roi" in request_lower:
            tasks.append(Task(
                task_id=f"TASK-{uuid.uuid4().hex[:8]}",
                task_type="cost_analysis",
                description="Analyze costs and financial impact",
                priority=priority,
                inputs={"request": request, "context": context}
            ))
        
        # Emergency-related tasks
        if "emergency" in request_lower or "urgent" in request_lower:
            tasks.append(Task(
                task_id=f"TASK-{uuid.uuid4().hex[:8]}",
                task_type="emergency_response",
                description="Coordinate emergency response",
                priority=TaskPriority.CRITICAL,
                inputs={"request": request, "context": context}
            ))
        
        # Inspection tasks
        if "inspect" in request_lower or "image" in request_lower:
            tasks.append(Task(
                task_id=f"TASK-{uuid.uuid4().hex[:8]}",
                task_type="inspection_analysis",
                description="Analyze inspection data",
                priority=priority,
                inputs={"request": request, "context": context}
            ))
        
        # If no specific tasks identified, create a general analysis task
        if not tasks:
            tasks.append(Task(
                task_id=f"TASK-{uuid.uuid4().hex[:8]}",
                task_type="general_analysis",
                description="General industrial analysis",
                priority=priority,
                inputs={"request": request, "context": context}
            ))
        
        return tasks
    
    async def _create_orchestration_plan(
        self,
        request_id: str,
        tasks: List[Task],
        mode: ExecutionMode
    ) -> OrchestrationPlan:
        """Create execution plan with task allocation"""
        plan_id = f"PLAN-{uuid.uuid4().hex[:8]}"
        
        # Assign agents to tasks based on capabilities
        for task in tasks:
            task.assigned_agent = self._select_best_agent(task)
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(tasks)
        
        # Determine execution order based on dependencies and mode
        execution_order = self._determine_execution_order(
            tasks,
            dependency_graph,
            mode
        )
        
        # Estimate duration and cost
        estimated_duration = self._estimate_duration(tasks, execution_order, mode)
        total_cost = self._estimate_cost(tasks)
        
        plan = OrchestrationPlan(
            plan_id=plan_id,
            request_id=request_id,
            execution_mode=mode,
            tasks=tasks,
            dependency_graph=dependency_graph,
            execution_order=execution_order,
            estimated_duration=estimated_duration,
            total_cost=total_cost
        )
        
        return plan
    
    def _select_best_agent(self, task: Task) -> str:
        """Select best agent for a task based on capabilities"""
        # Map task types to agent names
        task_agent_map = {
            "asset_analysis": "asset_health_agent",
            "maintenance_planning": "maintenance_agent",
            "safety_assessment": "safety_agent",
            "cost_analysis": "cost_agent",
            "emergency_response": "emergency_agent",
            "inspection_analysis": "inspection_agent",
            "general_analysis": "supervisor_agent"
        }
        
        agent_name = task_agent_map.get(task.task_type, "supervisor_agent")
        
        # Check if agent is registered
        if agent_name not in self.agents:
            agent_name = "supervisor_agent"
        
        return agent_name
    
    def _build_dependency_graph(self, tasks: List[Task]) -> Dict[str, List[str]]:
        """Build task dependency graph"""
        graph = defaultdict(list)
        
        # In this simplified version, we'll create some logical dependencies
        # Safety and asset analysis should happen first
        # Then maintenance and cost can run in parallel
        
        task_map = {t.task_id: t for t in tasks}
        
        for task in tasks:
            if task.task_type in ["maintenance_planning", "cost_analysis"]:
                # These depend on asset analysis if available
                for dep_task in tasks:
                    if dep_task.task_type == "asset_analysis":
                        task.dependencies.append(dep_task.task_id)
                        graph[dep_task.task_id].append(task.task_id)
        
        return dict(graph)
    
    def _determine_execution_order(
        self,
        tasks: List[Task],
        dependency_graph: Dict[str, List[str]],
        mode: ExecutionMode
    ) -> List[List[str]]:
        """Determine execution order (stages of parallel execution)"""
        
        if mode == ExecutionMode.SEQUENTIAL:
            # All tasks run one by one
            return [[task.task_id] for task in tasks]
        
        elif mode == ExecutionMode.PARALLEL:
            # Group tasks by dependencies (topological sort by levels)
            return self._topological_sort_levels(tasks, dependency_graph)
        
        else:  # HIERARCHICAL or CONDITIONAL
            # Supervisor first, then workers
            stages = []
            supervisor_tasks = [t for t in tasks if "supervisor" in t.assigned_agent.lower()]
            worker_tasks = [t for t in tasks if "supervisor" not in t.assigned_agent.lower()]
            
            if supervisor_tasks:
                stages.append([t.task_id for t in supervisor_tasks])
            if worker_tasks:
                stages.append([t.task_id for t in worker_tasks])
            
            return stages if stages else [[t.task_id for t in tasks]]
    
    def _topological_sort_levels(
        self,
        tasks: List[Task],
        dependency_graph: Dict[str, List[str]]
    ) -> List[List[str]]:
        """Topological sort returning levels for parallel execution"""
        task_map = {t.task_id: t for t in tasks}
        in_degree = {t.task_id: len(t.dependencies) for t in tasks}
        
        levels = []
        current_level = [tid for tid, degree in in_degree.items() if degree == 0]
        
        while current_level:
            levels.append(current_level)
            next_level = []
            
            for task_id in current_level:
                for dependent in dependency_graph.get(task_id, []):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_level.append(dependent)
            
            current_level = next_level
        
        return levels
    
    def _estimate_duration(
        self,
        tasks: List[Task],
        execution_order: List[List[str]],
        mode: ExecutionMode
    ) -> float:
        """Estimate total execution duration"""
        task_map = {t.task_id: t for t in tasks}
        total_duration = 0.0
        
        for stage in execution_order:
            # In parallel execution, stage duration is max of all tasks in stage
            stage_durations = []
            for task_id in stage:
                task = task_map[task_id]
                agent = self.agents.get(task.assigned_agent)
                if agent:
                    stage_durations.append(agent.average_execution_time)
                else:
                    stage_durations.append(5.0)  # Default
            
            total_duration += max(stage_durations) if stage_durations else 0.0
        
        return total_duration
    
    def _estimate_cost(self, tasks: List[Task]) -> float:
        """Estimate total execution cost"""
        total_cost = 0.0
        for task in tasks:
            agent = self.agents.get(task.assigned_agent)
            if agent:
                total_cost += agent.cost_per_execution
        return total_cost
    
    async def _execute_plan(self, plan: OrchestrationPlan) -> List[AgentExecution]:
        """Execute orchestration plan"""
        executions = []
        task_map = {t.task_id: t for t in plan.tasks}
        
        # Execute stage by stage
        for stage_idx, stage in enumerate(plan.execution_order):
            print(f"   📍 Stage {stage_idx + 1}/{len(plan.execution_order)}: {len(stage)} tasks")
            
            # Execute all tasks in this stage in parallel
            stage_executions = await asyncio.gather(*[
                self._execute_task(task_map[task_id])
                for task_id in stage
            ])
            
            executions.extend(stage_executions)
        
        return executions
    
    async def _execute_task(self, task: Task) -> AgentExecution:
        """Execute a single task with an agent"""
        execution_id = f"EXEC-{uuid.uuid4().hex[:8]}"
        start_time = datetime.utcnow()
        
        task.status = AgentStatus.RUNNING
        task.started_at = start_time
        
        execution = AgentExecution(
            execution_id=execution_id,
            agent_name=task.assigned_agent,
            task=task,
            status=AgentStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # Get agent instance
            agent = self.agent_instances.get(task.assigned_agent)
            
            if not agent:
                raise ValueError(f"Agent not found: {task.assigned_agent}")
            
            # Check cache if enabled
            cache_key = self._get_cache_key(task)
            if self.enable_caching and cache_key in self.cache:
                result = self.cache[cache_key]
                print(f"      💾 Cache hit for {task.assigned_agent}")
            else:
                # Execute agent
                print(f"      ⚙️  Executing {task.assigned_agent}...")
                result = await self._call_agent(agent, task)
                
                # Cache result
                if self.enable_caching:
                    self.cache[cache_key] = result
            
            # Mark as completed
            task.status = AgentStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.outputs = result
            
            execution.status = AgentStatus.COMPLETED
            execution.result = result
            
        except asyncio.TimeoutError:
            task.status = AgentStatus.TIMEOUT
            execution.status = AgentStatus.TIMEOUT
            execution.error = "Task execution timeout"
            print(f"      ⏱️  Timeout for {task.assigned_agent}")
            
        except Exception as e:
            task.status = AgentStatus.FAILED
            task.error = str(e)
            execution.status = AgentStatus.FAILED
            execution.error = str(e)
            print(f"      ❌ Error in {task.assigned_agent}: {str(e)}")
            
            # Retry logic
            if task.retries < task.max_retries:
                task.retries += 1
                print(f"      🔄 Retrying {task.assigned_agent} (attempt {task.retries}/{task.max_retries})")
                return await self._execute_task(task)
        
        finally:
            end_time = datetime.utcnow()
            execution.end_time = end_time
            execution.duration_seconds = (end_time - start_time).total_seconds()
            
            # Update metrics
            self._update_metrics(task.assigned_agent, execution)
        
        return execution
    
    async def _call_agent(self, agent: Any, task: Task) -> Dict[str, Any]:
        """Call agent with timeout"""
        try:
            # Simulate agent execution with timeout
            result = await asyncio.wait_for(
                self._agent_execution_wrapper(agent, task),
                timeout=self.default_timeout
            )
            return result
        except asyncio.TimeoutError:
            raise
    
    async def _agent_execution_wrapper(self, agent: Any, task: Task) -> Dict[str, Any]:
        """Wrapper for agent execution"""
        # This would call the actual agent method
        # For now, simulate with a delay and mock response
        await asyncio.sleep(0.5)  # Simulate processing
        
        return {
            "agent": task.assigned_agent,
            "task_type": task.task_type,
            "status": "completed",
            "analysis": f"Analysis completed by {task.assigned_agent}",
            "confidence": 0.85,
            "recommendations": [
                f"Recommendation 1 from {task.assigned_agent}",
                f"Recommendation 2 from {task.assigned_agent}"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_cache_key(self, task: Task) -> str:
        """Generate cache key for task"""
        # Simple cache key based on task type and inputs
        inputs_str = json.dumps(task.inputs, sort_keys=True)
        return f"{task.task_type}:{hash(inputs_str)}"
    
    def _update_metrics(self, agent_name: str, execution: AgentExecution):
        """Update agent performance metrics"""
        metrics = self.agent_metrics[agent_name]
        
        # Update average execution time
        if "total_executions" not in metrics:
            metrics["total_executions"] = 0
            metrics["total_duration"] = 0.0
            metrics["success_count"] = 0
            metrics["failure_count"] = 0
        
        metrics["total_executions"] += 1
        metrics["total_duration"] += execution.duration_seconds
        
        if execution.status == AgentStatus.COMPLETED:
            metrics["success_count"] += 1
        else:
            metrics["failure_count"] += 1
        
        metrics["avg_duration"] = metrics["total_duration"] / metrics["total_executions"]
        metrics["success_rate"] = metrics["success_count"] / metrics["total_executions"]
    
    async def _aggregate_results(
        self,
        executions: List[AgentExecution],
        original_request: str
    ) -> Dict[str, Any]:
        """Aggregate results from multiple agents"""
        aggregated = {
            "request": original_request,
            "timestamp": datetime.utcnow().isoformat(),
            "agents_executed": len(executions),
            "results_by_agent": {},
            "combined_analysis": {},
            "all_recommendations": []
        }
        
        for execution in executions:
            if execution.status == AgentStatus.COMPLETED and execution.result:
                aggregated["results_by_agent"][execution.agent_name] = execution.result
                
                # Collect recommendations
                if "recommendations" in execution.result:
                    aggregated["all_recommendations"].extend(
                        execution.result["recommendations"]
                    )
        
        return aggregated
    
    async def _resolve_conflicts(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts between agent results"""
        # Simple conflict resolution: prioritize by agent reliability
        # In production, use more sophisticated methods
        
        # Remove duplicate recommendations
        if "all_recommendations" in aggregated:
            aggregated["all_recommendations"] = list(set(
                aggregated["all_recommendations"]
            ))
        
        return aggregated
    
    def get_agent_metrics(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for agents"""
        if agent_name:
            return self.agent_metrics.get(agent_name, {})
        return dict(self.agent_metrics)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "registered_agents": len(self.agents),
            "active_executions": len(self.active_executions),
            "cache_size": len(self.cache),
            "total_executions": sum(
                m.get("total_executions", 0)
                for m in self.agent_metrics.values()
            ),
            "avg_success_rate": sum(
                m.get("success_rate", 0)
                for m in self.agent_metrics.values()
            ) / len(self.agent_metrics) if self.agent_metrics else 0.0
        }


# Export
__all__ = [
    "MultiAgentOrchestrator",
    "AgentCapability",
    "Task",
    "OrchestrationResult",
    "ExecutionMode",
    "TaskPriority",
    "AgentStatus"
]
