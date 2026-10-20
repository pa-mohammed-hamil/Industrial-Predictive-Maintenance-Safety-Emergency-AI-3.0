"""
Digital Twin Engine
Provides real-time virtual replicas of physical assets with physics simulation,
state synchronization, and predictive modeling.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from dataclasses import dataclass, field
import numpy as np
import json


class TwinType(Enum):
    """Types of digital twins"""
    COMPONENT = "component"  # Individual component (motor, pump, valve)
    ASSET = "asset"  # Complete asset (machine, equipment)
    SYSTEM = "system"  # System of assets (production line)
    PROCESS = "process"  # Business process
    FACILITY = "facility"  # Entire facility


class SimulationMode(Enum):
    """Physics simulation modes"""
    REAL_TIME = "real_time"  # Real-time sync with physical asset
    ACCELERATED = "accelerated"  # Faster than real-time for predictions
    HISTORICAL = "historical"  # Replay historical scenarios
    WHAT_IF = "what_if"  # Scenario testing


class TwinState(Enum):
    """Digital twin lifecycle states"""
    INITIALIZING = "initializing"
    SYNCHRONIZED = "synchronized"
    DESYNCHRONIZED = "desynchronized"
    SIMULATING = "simulating"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class PhysicsModel:
    """Physics-based model for asset behavior"""
    model_type: str  # "thermal", "mechanical", "electrical", "fluid", "vibration"
    equations: Dict[str, str]  # Variable: equation mapping
    parameters: Dict[str, float]  # Physical parameters
    state_variables: Dict[str, float]  # Current state
    constraints: Dict[str, Tuple[float, float]]  # Min/max bounds
    update_frequency_hz: float = 10.0
    
    def simulate_step(self, dt: float, inputs: Dict[str, float]) -> Dict[str, float]:
        """Simulate one physics timestep"""
        # Simplified physics simulation - in production, use FEA/CFD engines
        new_state = self.state_variables.copy()
        
        if self.model_type == "thermal":
            # Heat transfer: dT/dt = (Q_in - Q_out) / (m * c_p)
            temp = self.state_variables.get("temperature", 25.0)
            heat_in = inputs.get("heat_input", 0.0)
            heat_out = self.parameters["heat_transfer_coeff"] * (temp - inputs.get("ambient_temp", 25.0))
            thermal_mass = self.parameters["mass"] * self.parameters["specific_heat"]
            
            dT = (heat_in - heat_out) / thermal_mass * dt
            new_state["temperature"] = temp + dT
            
        elif self.model_type == "mechanical":
            # Rotational dynamics: dω/dt = (τ - b*ω) / I
            omega = self.state_variables.get("angular_velocity", 0.0)
            torque = inputs.get("torque", 0.0)
            damping = self.parameters["damping_coefficient"]
            inertia = self.parameters["moment_of_inertia"]
            
            d_omega = (torque - damping * omega) / inertia * dt
            new_state["angular_velocity"] = omega + d_omega
            new_state["rotation_angle"] = self.state_variables.get("rotation_angle", 0.0) + omega * dt
            
        elif self.model_type == "electrical":
            # RL circuit: di/dt = (V - R*i) / L
            current = self.state_variables.get("current", 0.0)
            voltage = inputs.get("voltage", 0.0)
            resistance = self.parameters["resistance"]
            inductance = self.parameters["inductance"]
            
            di = (voltage - resistance * current) / inductance * dt
            new_state["current"] = current + di
            new_state["power"] = voltage * new_state["current"]
            
        elif self.model_type == "vibration":
            # Damped oscillator: d²x/dt² + 2ζω₀(dx/dt) + ω₀²x = F/m
            position = self.state_variables.get("position", 0.0)
            velocity = self.state_variables.get("velocity", 0.0)
            force = inputs.get("force", 0.0)
            
            omega_0 = self.parameters["natural_frequency"]
            zeta = self.parameters["damping_ratio"]
            mass = self.parameters["mass"]
            
            acceleration = (force / mass) - (2 * zeta * omega_0 * velocity) - (omega_0**2 * position)
            new_state["velocity"] = velocity + acceleration * dt
            new_state["position"] = position + velocity * dt
            new_state["acceleration"] = acceleration
        
        # Apply constraints
        for var, (min_val, max_val) in self.constraints.items():
            if var in new_state:
                new_state[var] = np.clip(new_state[var], min_val, max_val)
        
        return new_state


@dataclass
class DigitalTwin:
    """Complete digital twin of a physical asset"""
    twin_id: str
    asset_id: str
    twin_type: TwinType
    name: str
    description: str
    
    # Physics models
    physics_models: List[PhysicsModel] = field(default_factory=list)
    
    # State tracking
    state: TwinState = TwinState.INITIALIZING
    last_sync_time: Optional[datetime] = None
    physical_state: Dict[str, Any] = field(default_factory=dict)
    virtual_state: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    sync_accuracy: float = 0.0  # 0-100%
    prediction_horizon_hours: float = 24.0
    simulation_speed: float = 1.0  # Multiplier (1.0 = real-time)
    
    # Configuration
    sync_interval_seconds: float = 1.0
    state_history_size: int = 10000
    
    # History
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class DigitalTwinEngine:
    """
    Digital Twin Engine
    
    Manages lifecycle of digital twins with real-time synchronization,
    physics simulation, and predictive analytics.
    """
    
    def __init__(self):
        self.twins: Dict[str, DigitalTwin] = {}
        self.simulation_tasks: Dict[str, asyncio.Task] = {}
        self.sync_tasks: Dict[str, asyncio.Task] = {}
        
    async def create_twin(
        self,
        asset_id: str,
        twin_type: TwinType,
        name: str,
        description: str,
        physics_models: Optional[List[Dict[str, Any]]] = None
    ) -> DigitalTwin:
        """Create a new digital twin"""
        twin_id = f"twin_{asset_id}_{datetime.utcnow().timestamp()}"
        
        # Initialize physics models
        models = []
        if physics_models:
            for model_def in physics_models:
                model = PhysicsModel(
                    model_type=model_def["model_type"],
                    equations=model_def.get("equations", {}),
                    parameters=model_def.get("parameters", {}),
                    state_variables=model_def.get("initial_state", {}),
                    constraints=model_def.get("constraints", {}),
                    update_frequency_hz=model_def.get("update_frequency_hz", 10.0)
                )
                models.append(model)
        
        twin = DigitalTwin(
            twin_id=twin_id,
            asset_id=asset_id,
            twin_type=twin_type,
            name=name,
            description=description,
            physics_models=models,
            state=TwinState.INITIALIZING
        )
        
        self.twins[twin_id] = twin
        
        # Start synchronization
        await self._initialize_twin(twin)
        
        return twin
    
    async def _initialize_twin(self, twin: DigitalTwin):
        """Initialize twin and start synchronization"""
        # Fetch initial state from physical asset
        physical_state = await self._fetch_physical_state(twin.asset_id)
        twin.physical_state = physical_state
        twin.virtual_state = physical_state.copy()
        twin.last_sync_time = datetime.utcnow()
        twin.state = TwinState.SYNCHRONIZED
        
        # Initialize physics model states
        for model in twin.physics_models:
            for var in model.state_variables:
                if var in physical_state:
                    model.state_variables[var] = physical_state[var]
        
        # Start background sync task
        self.sync_tasks[twin.twin_id] = asyncio.create_task(
            self._sync_loop(twin)
        )
    
    async def _sync_loop(self, twin: DigitalTwin):
        """Continuous synchronization with physical asset"""
        while twin.twin_id in self.twins:
            try:
                # Fetch latest physical state
                physical_state = await self._fetch_physical_state(twin.asset_id)
                twin.physical_state = physical_state
                twin.last_sync_time = datetime.utcnow()
                
                # Calculate sync accuracy
                twin.sync_accuracy = self._calculate_sync_accuracy(
                    twin.virtual_state,
                    physical_state
                )
                
                # Update virtual state if desynchronized
                if twin.sync_accuracy < 90.0:
                    twin.state = TwinState.DESYNCHRONIZED
                    await self._resync_twin(twin)
                else:
                    twin.state = TwinState.SYNCHRONIZED
                
                # Store history
                twin.state_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "physical": physical_state,
                    "virtual": twin.virtual_state.copy(),
                    "sync_accuracy": twin.sync_accuracy
                })
                
                # Limit history size
                if len(twin.state_history) > twin.state_history_size:
                    twin.state_history = twin.state_history[-twin.state_history_size:]
                
                twin.updated_at = datetime.utcnow()
                
                await asyncio.sleep(twin.sync_interval_seconds)
                
            except Exception as e:
                twin.state = TwinState.ERROR
                print(f"Sync error for twin {twin.twin_id}: {e}")
                await asyncio.sleep(5.0)  # Retry after delay
    
    async def _fetch_physical_state(self, asset_id: str) -> Dict[str, Any]:
        """Fetch current state from physical asset (IoT sensors)"""
        # Simulate fetching from IoT platform
        await asyncio.sleep(0.01)  # Simulate network delay
        
        # In production, fetch from actual sensors/SCADA
        return {
            "temperature": np.random.normal(75.0, 5.0),
            "vibration": np.random.normal(2.5, 0.5),
            "pressure": np.random.normal(100.0, 10.0),
            "speed_rpm": np.random.normal(1500, 50),
            "current": np.random.normal(15.0, 2.0),
            "voltage": np.random.normal(240.0, 5.0),
            "power_kw": np.random.normal(3.6, 0.5),
            "efficiency": np.random.uniform(0.85, 0.95),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_sync_accuracy(
        self,
        virtual_state: Dict[str, Any],
        physical_state: Dict[str, Any]
    ) -> float:
        """Calculate synchronization accuracy percentage"""
        if not virtual_state or not physical_state:
            return 0.0
        
        common_keys = set(virtual_state.keys()) & set(physical_state.keys())
        if not common_keys:
            return 0.0
        
        errors = []
        for key in common_keys:
            if isinstance(virtual_state[key], (int, float)) and isinstance(physical_state[key], (int, float)):
                virtual_val = float(virtual_state[key])
                physical_val = float(physical_state[key])
                
                if physical_val != 0:
                    relative_error = abs(virtual_val - physical_val) / abs(physical_val)
                    errors.append(relative_error)
        
        if not errors:
            return 100.0
        
        avg_error = np.mean(errors)
        accuracy = max(0.0, (1.0 - avg_error) * 100.0)
        return min(100.0, accuracy)
    
    async def _resync_twin(self, twin: DigitalTwin):
        """Resynchronize virtual state with physical state"""
        twin.virtual_state = twin.physical_state.copy()
        
        # Update physics model states
        for model in twin.physics_models:
            for var in model.state_variables:
                if var in twin.physical_state:
                    model.state_variables[var] = twin.physical_state[var]
    
    async def simulate(
        self,
        twin_id: str,
        duration_hours: float,
        mode: SimulationMode = SimulationMode.ACCELERATED,
        scenario: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Run physics simulation"""
        twin = self.twins.get(twin_id)
        if not twin:
            raise ValueError(f"Twin {twin_id} not found")
        
        twin.state = TwinState.SIMULATING
        
        # Simulation parameters
        dt = 0.1  # 100ms timestep
        speed_multiplier = 100.0 if mode == SimulationMode.ACCELERATED else 1.0
        total_steps = int((duration_hours * 3600) / dt)
        
        simulation_results = []
        current_state = twin.virtual_state.copy()
        
        for step in range(total_steps):
            timestamp = twin.last_sync_time + timedelta(seconds=step * dt)
            
            # Prepare inputs for physics models
            inputs = scenario.copy() if scenario else {}
            inputs["ambient_temp"] = 25.0
            inputs["load_factor"] = 0.75
            
            # Run each physics model
            for model in twin.physics_models:
                new_state = model.simulate_step(dt, inputs)
                model.state_variables.update(new_state)
                current_state.update(new_state)
            
            # Sample results (not every timestep)
            if step % 100 == 0:
                simulation_results.append({
                    "timestamp": timestamp.isoformat(),
                    "state": current_state.copy(),
                    "step": step
                })
        
        twin.predictions = simulation_results
        twin.state = TwinState.SYNCHRONIZED
        
        return simulation_results
    
    async def predict_failure(
        self,
        twin_id: str,
        prediction_horizon_hours: float = 24.0
    ) -> Dict[str, Any]:
        """Predict future failures using digital twin"""
        twin = self.twins.get(twin_id)
        if not twin:
            raise ValueError(f"Twin {twin_id} not found")
        
        # Run simulation to predict future state
        future_states = await self.simulate(
            twin_id,
            prediction_horizon_hours,
            SimulationMode.ACCELERATED
        )
        
        # Analyze for failure conditions
        failures_predicted = []
        
        for state_data in future_states:
            state = state_data["state"]
            timestamp = state_data["timestamp"]
            
            # Check threshold violations
            if state.get("temperature", 0) > 150.0:
                failures_predicted.append({
                    "timestamp": timestamp,
                    "failure_type": "overheating",
                    "severity": "critical",
                    "predicted_value": state["temperature"],
                    "threshold": 150.0
                })
            
            if state.get("vibration", 0) > 10.0:
                failures_predicted.append({
                    "timestamp": timestamp,
                    "failure_type": "excessive_vibration",
                    "severity": "high",
                    "predicted_value": state["vibration"],
                    "threshold": 10.0
                })
            
            if state.get("efficiency", 1.0) < 0.6:
                failures_predicted.append({
                    "timestamp": timestamp,
                    "failure_type": "efficiency_degradation",
                    "severity": "medium",
                    "predicted_value": state["efficiency"],
                    "threshold": 0.6
                })
        
        return {
            "twin_id": twin_id,
            "prediction_horizon_hours": prediction_horizon_hours,
            "failures_predicted": failures_predicted,
            "failure_probability": min(len(failures_predicted) / 10.0, 1.0),
            "recommended_action": "Schedule preventive maintenance" if failures_predicted else "Continue normal operation",
            "predicted_at": datetime.utcnow().isoformat()
        }
    
    async def optimize_parameters(
        self,
        twin_id: str,
        objective: str,  # "minimize_energy", "maximize_efficiency", "extend_lifetime"
        constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """Optimize operational parameters using digital twin"""
        twin = self.twins.get(twin_id)
        if not twin:
            raise ValueError(f"Twin {twin_id} not found")
        
        # Simplified optimization - in production, use scipy.optimize or genetic algorithms
        best_params = {}
        best_score = float('-inf') if "maximize" in objective else float('inf')
        
        # Test parameter combinations
        param_ranges = constraints or {
            "speed_rpm": (1000, 2000),
            "load_factor": (0.5, 1.0),
            "cooling_rate": (0.5, 2.0)
        }
        
        for _ in range(20):  # Sample 20 combinations
            test_params = {
                key: np.random.uniform(low, high)
                for key, (low, high) in param_ranges.items()
            }
            
            # Simulate with these parameters
            results = await self.simulate(
                twin_id,
                1.0,  # 1 hour simulation
                SimulationMode.ACCELERATED,
                test_params
            )
            
            # Calculate objective score
            final_state = results[-1]["state"]
            
            if objective == "minimize_energy":
                score = -final_state.get("power_kw", 0)
            elif objective == "maximize_efficiency":
                score = final_state.get("efficiency", 0)
            elif objective == "extend_lifetime":
                # Lower temperature and vibration = longer life
                score = -(final_state.get("temperature", 0) / 100.0 + final_state.get("vibration", 0))
            else:
                score = 0
            
            # Update best
            is_better = (score > best_score) if "maximize" in objective else (score < best_score)
            if is_better:
                best_score = score
                best_params = test_params.copy()
        
        return {
            "twin_id": twin_id,
            "objective": objective,
            "optimized_parameters": best_params,
            "expected_improvement": abs(best_score),
            "optimization_time": datetime.utcnow().isoformat()
        }
    
    def get_twin(self, twin_id: str) -> Optional[DigitalTwin]:
        """Get digital twin by ID"""
        return self.twins.get(twin_id)
    
    def list_twins(
        self,
        twin_type: Optional[TwinType] = None,
        state: Optional[TwinState] = None
    ) -> List[DigitalTwin]:
        """List all digital twins with optional filters"""
        twins = list(self.twins.values())
        
        if twin_type:
            twins = [t for t in twins if t.twin_type == twin_type]
        
        if state:
            twins = [t for t in twins if t.state == state]
        
        return twins
    
    async def delete_twin(self, twin_id: str):
        """Delete a digital twin"""
        if twin_id in self.sync_tasks:
            self.sync_tasks[twin_id].cancel()
            del self.sync_tasks[twin_id]
        
        if twin_id in self.simulation_tasks:
            self.simulation_tasks[twin_id].cancel()
            del self.simulation_tasks[twin_id]
        
        if twin_id in self.twins:
            del self.twins[twin_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        twins = list(self.twins.values())
        
        return {
            "total_twins": len(twins),
            "by_type": {
                twin_type.value: len([t for t in twins if t.twin_type == twin_type])
                for twin_type in TwinType
            },
            "by_state": {
                state.value: len([t for t in twins if t.state == state])
                for state in TwinState
            },
            "average_sync_accuracy": np.mean([t.sync_accuracy for t in twins]) if twins else 0.0,
            "synchronized_twins": len([t for t in twins if t.state == TwinState.SYNCHRONIZED]),
            "active_simulations": len(self.simulation_tasks)
        }


# Global instance
digital_twin_engine = DigitalTwinEngine()
