# app/integrations/plc/simulator.py

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator

from .adapter import (
    MemoryPLCAdapter,
    PLCAccess,
    PLCDataType,
    PLCTag,
)

logger = logging.getLogger(__name__)


class MachineMode(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAULT = "fault"
    EMERGENCY_STOP = "emergency_stop"


class FailureMode(str, Enum):
    NONE = "none"
    BEARING_WEAR = "bearing_wear"
    MOTOR_OVERHEAT = "motor_overheat"
    VIBRATION = "vibration"
    OIL_PRESSURE = "oil_pressure"
    OVERCURRENT = "overcurrent"
    SENSOR_FAILURE = "sensor_failure"


@dataclass(slots=True)
class SimulatorConfig:
    asset_id: str = "MACHINE-001"
    site_id: str = "PLANT-001"

    interval_seconds: float = 1.0

    nominal_speed_rpm: float = 1450.0
    nominal_load_percent: float = 70.0

    ambient_temperature_c: float = 28.0

    random_seed: int | None = None


@dataclass(slots=True)
class SimulationState:
    mode: MachineMode = MachineMode.STOPPED

    operating_hours: float = 0.0
    production_count: int = 0
    scrap_count: int = 0

    degradation: float = 0.0

    failure_mode: FailureMode = FailureMode.NONE

    fire_detected: bool = False
    smoke_detected: bool = False
    emergency_stop: bool = False


# ---------------------------------------------------------
# PLC tags
# ---------------------------------------------------------

MACHINE_RUNNING = PLCTag(
    name="machine_running",
    address="DB1.DBX0.0",
    data_type=PLCDataType.BOOL,
)

MACHINE_SPEED = PLCTag(
    name="machine_speed",
    address="DB1.DBD2",
    data_type=PLCDataType.FLOAT,
    unit="rpm",
    minimum=0.0,
    maximum=5000.0,
)

MACHINE_LOAD = PLCTag(
    name="machine_load",
    address="DB1.DBD6",
    data_type=PLCDataType.FLOAT,
    unit="%",
    minimum=0.0,
    maximum=100.0,
)

OPERATING_HOURS = PLCTag(
    name="operating_hours",
    address="DB1.DBD10",
    data_type=PLCDataType.FLOAT,
    unit="hours",
)

BEARING_TEMPERATURE = PLCTag(
    name="bearing_temperature",
    address="DB2.DBD0",
    data_type=PLCDataType.FLOAT,
    unit="°C",
)

MOTOR_TEMPERATURE = PLCTag(
    name="motor_temperature",
    address="DB2.DBD4",
    data_type=PLCDataType.FLOAT,
    unit="°C",
)

AMBIENT_TEMPERATURE = PLCTag(
    name="ambient_temperature",
    address="DB2.DBD8",
    data_type=PLCDataType.FLOAT,
    unit="°C",
)

VIBRATION_RMS = PLCTag(
    name="vibration_rms",
    address="DB3.DBD0",
    data_type=PLCDataType.FLOAT,
    unit="mm/s",
)

VIBRATION_X = PLCTag(
    name="vibration_x",
    address="DB3.DBD4",
    data_type=PLCDataType.FLOAT,
    unit="mm/s",
)

VIBRATION_Y = PLCTag(
    name="vibration_y",
    address="DB3.DBD8",
    data_type=PLCDataType.FLOAT,
    unit="mm/s",
)

VIBRATION_Z = PLCTag(
    name="vibration_z",
    address="DB3.DBD12",
    data_type=PLCDataType.FLOAT,
    unit="mm/s",
)

OIL_PRESSURE = PLCTag(
    name="oil_pressure",
    address="DB4.DBD0",
    data_type=PLCDataType.FLOAT,
    unit="bar",
)

INLET_PRESSURE = PLCTag(
    name="inlet_pressure",
    address="DB4.DBD4",
    data_type=PLCDataType.FLOAT,
    unit="bar",
)

OUTLET_PRESSURE = PLCTag(
    name="outlet_pressure",
    address="DB4.DBD8",
    data_type=PLCDataType.FLOAT,
    unit="bar",
)

CURRENT = PLCTag(
    name="current",
    address="DB5.DBD0",
    data_type=PLCDataType.FLOAT,
    unit="A",
)

VOLTAGE = PLCTag(
    name="voltage",
    address="DB5.DBD4",
    data_type=PLCDataType.FLOAT,
    unit="V",
)

POWER = PLCTag(
    name="power",
    address="DB5.DBD8",
    data_type=PLCDataType.FLOAT,
    unit="kW",
)

ENERGY = PLCTag(
    name="energy_consumption",
    address="DB5.DBD12",
    data_type=PLCDataType.FLOAT,
    unit="kWh",
)

PRODUCTION_COUNT = PLCTag(
    name="production_count",
    address="DB6.DBD0",
    data_type=PLCDataType.INT,
)

SCRAP_COUNT = PLCTag(
    name="scrap_count",
    address="DB6.DBD4",
    data_type=PLCDataType.INT,
)

FIRE_DETECTED = PLCTag(
    name="fire_detected",
    address="DB7.DBX0.0",
    data_type=PLCDataType.BOOL,
)

SMOKE_DETECTED = PLCTag(
    name="smoke_detected",
    address="DB7.DBX0.1",
    data_type=PLCDataType.BOOL,
)

OVERHEATING = PLCTag(
    name="overheating",
    address="DB7.DBX0.2",
    data_type=PLCDataType.BOOL,
)

EMERGENCY_STOP = PLCTag(
    name="emergency_stop",
    address="DB7.DBX0.3",
    data_type=PLCDataType.BOOL,
)

HEALTH_SCORE = PLCTag(
    name="health_score",
    address="DB8.DBD0",
    data_type=PLCDataType.FLOAT,
    unit="%",
)

FAILURE_PROBABILITY = PLCTag(
    name="failure_probability",
    address="DB8.DBD4",
    data_type=PLCDataType.FLOAT,
)

RUL_HOURS = PLCTag(
    name="rul_hours",
    address="DB8.DBD8",
    data_type=PLCDataType.FLOAT,
    unit="hours",
)

START_COMMAND = PLCTag(
    name="start_command",
    address="DB10.DBX0.0",
    data_type=PLCDataType.BOOL,
    access=PLCAccess.READ_WRITE,
)

STOP_COMMAND = PLCTag(
    name="stop_command",
    address="DB10.DBX0.1",
    data_type=PLCDataType.BOOL,
    access=PLCAccess.READ_WRITE,
)

RESET_COMMAND = PLCTag(
    name="reset_command",
    address="DB10.DBX0.2",
    data_type=PLCDataType.BOOL,
    access=PLCAccess.READ_WRITE,
)


SIMULATOR_TAGS: tuple[PLCTag, ...] = (
    MACHINE_RUNNING,
    MACHINE_SPEED,
    MACHINE_LOAD,
    OPERATING_HOURS,
    BEARING_TEMPERATURE,
    MOTOR_TEMPERATURE,
    AMBIENT_TEMPERATURE,
    VIBRATION_RMS,
    VIBRATION_X,
    VIBRATION_Y,
    VIBRATION_Z,
    OIL_PRESSURE,
    INLET_PRESSURE,
    OUTLET_PRESSURE,
    CURRENT,
    VOLTAGE,
    POWER,
    ENERGY,
    PRODUCTION_COUNT,
    SCRAP_COUNT,
    FIRE_DETECTED,
    SMOKE_DETECTED,
    OVERHEATING,
    EMERGENCY_STOP,
    HEALTH_SCORE,
    FAILURE_PROBABILITY,
    RUL_HOURS,
)


class IndustrialPLCSimulator(MemoryPLCAdapter):
    """
    Realistic industrial PLC simulator.

    Simulates:
    - machine speed/load
    - temperatures
    - vibration
    - pressure
    - current/voltage/power
    - energy consumption
    - production
    - gradual degradation
    - health score
    - failure probability
    - RUL
    - fire/smoke/overheating
    - failure injection
    - emergency stop
    """

    def __init__(
        self,
        config: SimulatorConfig | None = None,
    ) -> None:
        self.config = config or SimulatorConfig()

        self.state = SimulationState()

        self._rng = random.Random(
            self.config.random_seed
        )

        self._simulation_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

        self._energy_kwh = 0.0
        self._simulation_started = time.monotonic()

        super().__init__(
            name=f"plc-{self.config.asset_id}",
            initial_values=self._initial_memory(),
        )

    def _initial_memory(self) -> dict[str, Any]:
        return {
            MACHINE_RUNNING.address: False,
            MACHINE_SPEED.address: 0.0,
            MACHINE_LOAD.address: 0.0,
            OPERATING_HOURS.address: 0.0,

            BEARING_TEMPERATURE.address: (
                self.config.ambient_temperature_c
            ),
            MOTOR_TEMPERATURE.address: (
                self.config.ambient_temperature_c
            ),
            AMBIENT_TEMPERATURE.address: (
                self.config.ambient_temperature_c
            ),

            VIBRATION_RMS.address: 0.1,
            VIBRATION_X.address: 0.05,
            VIBRATION_Y.address: 0.05,
            VIBRATION_Z.address: 0.05,

            OIL_PRESSURE.address: 0.0,
            INLET_PRESSURE.address: 0.0,
            OUTLET_PRESSURE.address: 0.0,

            CURRENT.address: 0.0,
            VOLTAGE.address: 415.0,
            POWER.address: 0.0,
            ENERGY.address: 0.0,

            PRODUCTION_COUNT.address: 0,
            SCRAP_COUNT.address: 0,

            FIRE_DETECTED.address: False,
            SMOKE_DETECTED.address: False,
            OVERHEATING.address: False,
            EMERGENCY_STOP.address: False,

            HEALTH_SCORE.address: 100.0,
            FAILURE_PROBABILITY.address: 0.01,
            RUL_HOURS.address: 5000.0,

            START_COMMAND.address: False,
            STOP_COMMAND.address: False,
            RESET_COMMAND.address: False,
        }

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    async def start_simulation(self) -> None:
        if self._simulation_task:
            if not self._simulation_task.done():
                return

        if not self.is_connected:
            await self.connect()

        self._stop_event.clear()

        self._simulation_task = asyncio.create_task(
            self._simulation_loop(),
            name=f"plc-simulator-{self.config.asset_id}",
        )

        logger.info(
            "PLC simulator started asset=%s",
            self.config.asset_id,
        )

    async def stop_simulation(self) -> None:
        self._stop_event.set()

        task = self._simulation_task

        if task:
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

        self._simulation_task = None

        logger.info(
            "PLC simulator stopped asset=%s",
            self.config.asset_id,
        )

    async def _simulation_loop(self) -> None:
        interval = max(
            0.05,
            self.config.interval_seconds,
        )

        while not self._stop_event.is_set():
            try:
                await self.step(interval)

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "PLC simulation step failed"
                )

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=interval,
                )
            except asyncio.TimeoutError:
                pass

    # --------------------------------------------------
    # Machine controls
    # --------------------------------------------------

    async def start_machine(self) -> None:
        if self.state.emergency_stop:
            raise RuntimeError(
                "Cannot start machine while "
                "emergency stop is active"
            )

        if self.state.mode == MachineMode.RUNNING:
            return

        self.state.mode = MachineMode.STARTING

        await asyncio.sleep(0.1)

        self.state.mode = MachineMode.RUNNING

        self._memory[MACHINE_RUNNING.address] = True
        self._memory[START_COMMAND.address] = False

        logger.info(
            "Simulator machine started asset=%s",
            self.config.asset_id,
        )

    async def stop_machine(self) -> None:
        self.state.mode = MachineMode.STOPPED

        self._memory[MACHINE_RUNNING.address] = False
        self._memory[MACHINE_SPEED.address] = 0.0
        self._memory[MACHINE_LOAD.address] = 0.0

        self._memory[CURRENT.address] = 0.0
        self._memory[POWER.address] = 0.0

        self._memory[STOP_COMMAND.address] = False

        logger.info(
            "Simulator machine stopped asset=%s",
            self.config.asset_id,
        )

    async def emergency_stop(self) -> None:
        self.state.emergency_stop = True
        self.state.mode = MachineMode.EMERGENCY_STOP

        self._memory[EMERGENCY_STOP.address] = True
        self._memory[MACHINE_RUNNING.address] = False
        self._memory[MACHINE_SPEED.address] = 0.0
        self._memory[MACHINE_LOAD.address] = 0.0
        self._memory[CURRENT.address] = 0.0
        self._memory[POWER.address] = 0.0

        logger.warning(
            "Emergency stop activated asset=%s",
            self.config.asset_id,
        )

    async def reset_emergency(self) -> None:
        self.state.emergency_stop = False

        self.state.fire_detected = False
        self.state.smoke_detected = False

        self.state.mode = MachineMode.STOPPED

        self._memory[EMERGENCY_STOP.address] = False
        self._memory[FIRE_DETECTED.address] = False
        self._memory[SMOKE_DETECTED.address] = False
        self._memory[OVERHEATING.address] = False
        self._memory[RESET_COMMAND.address] = False

        logger.info(
            "Emergency state reset asset=%s",
            self.config.asset_id,
        )

    # --------------------------------------------------
    # Command handling
    # --------------------------------------------------

    async def write_raw(
        self,
        address: str,
        value: Any,
    ) -> None:
        await super().write_raw(
            address,
            value,
        )

        if address == START_COMMAND.address and bool(value):
            await self.start_machine()

        elif address == STOP_COMMAND.address and bool(value):
            await self.stop_machine()

        elif address == RESET_COMMAND.address and bool(value):
            await self.reset_emergency()

    # --------------------------------------------------
    # Simulation
    # --------------------------------------------------

    async def step(
        self,
        elapsed_seconds: float | None = None,
    ) -> None:
        elapsed = (
            elapsed_seconds
            if elapsed_seconds is not None
            else self.config.interval_seconds
        )

        elapsed = max(
            0.001,
            float(elapsed),
        )

        ambient = (
            self.config.ambient_temperature_c
            + self._noise(0.4)
        )

        self._memory[
            AMBIENT_TEMPERATURE.address
        ] = round(ambient, 3)

        if self.state.mode not in {
            MachineMode.RUNNING,
            MachineMode.DEGRADED,
        }:
            self._simulate_idle(ambient)
            self._update_health()
            return

        self.state.operating_hours += (
            elapsed / 3600.0
        )

        self.state.degradation = min(
            1.0,
            self.state.degradation
            + elapsed / (3600.0 * 5000.0),
        )

        load = self._calculate_load()
        speed = self._calculate_speed(load)

        bearing_temp = self._bearing_temperature(
            ambient,
            load,
        )

        motor_temp = self._motor_temperature(
            ambient,
            load,
        )

        vibration = self._vibration(load)

        oil_pressure = self._oil_pressure(load)

        current = self._current(load)
        voltage = 415.0 + self._noise(3.0)

        power = max(
            0.0,
            (
                math.sqrt(3)
                * voltage
                * current
                * 0.88
            )
            / 1000.0,
        )

        self._energy_kwh += (
            power * elapsed / 3600.0
        )

        self._update_production(
            elapsed,
            load,
        )

        self._memory.update(
            {
                MACHINE_RUNNING.address: True,
                MACHINE_SPEED.address: round(
                    speed,
                    3,
                ),
                MACHINE_LOAD.address: round(
                    load,
                    3,
                ),
                OPERATING_HOURS.address: round(
                    self.state.operating_hours,
                    5,
                ),
                BEARING_TEMPERATURE.address: round(
                    bearing_temp,
                    3,
                ),
                MOTOR_TEMPERATURE.address: round(
                    motor_temp,
                    3,
                ),
                VIBRATION_RMS.address: round(
                    vibration,
                    4,
                ),
                VIBRATION_X.address: round(
                    vibration
                    * self._rng.uniform(
                        0.75,
                        1.15,
                    ),
                    4,
                ),
                VIBRATION_Y.address: round(
                    vibration
                    * self._rng.uniform(
                        0.70,
                        1.20,
                    ),
                    4,
                ),
                VIBRATION_Z.address: round(
                    vibration
                    * self._rng.uniform(
                        0.65,
                        1.10,
                    ),
                    4,
                ),
                OIL_PRESSURE.address: round(
                    oil_pressure,
                    3,
                ),
                INLET_PRESSURE.address: round(
                    4.0
                    + (load / 100.0) * 2.0
                    + self._noise(0.08),
                    3,
                ),
                OUTLET_PRESSURE.address: round(
                    6.0
                    + (load / 100.0) * 2.5
                    + self._noise(0.10),
                    3,
                ),
                CURRENT.address: round(
                    current,
                    3,
                ),
                VOLTAGE.address: round(
                    voltage,
                    3,
                ),
                POWER.address: round(
                    power,
                    3,
                ),
                ENERGY.address: round(
                    self._energy_kwh,
                    5,
                ),
                PRODUCTION_COUNT.address: (
                    self.state.production_count
                ),
                SCRAP_COUNT.address: (
                    self.state.scrap_count
                ),
            }
        )

        self._apply_failure_effects()
        self._evaluate_safety()
        self._update_health()

    def _simulate_idle(
        self,
        ambient: float,
    ) -> None:
        self._memory[MACHINE_RUNNING.address] = False

        current_speed = float(
            self._memory.get(
                MACHINE_SPEED.address,
                0.0,
            )
        )

        self._memory[MACHINE_SPEED.address] = max(
            0.0,
            current_speed * 0.5,
        )

        self._memory[MACHINE_LOAD.address] = 0.0

        self._memory[
            BEARING_TEMPERATURE.address
        ] = self._cool_towards(
            float(
                self._memory.get(
                    BEARING_TEMPERATURE.address,
                    ambient,
                )
            ),
            ambient,
        )

        self._memory[
            MOTOR_TEMPERATURE.address
        ] = self._cool_towards(
            float(
                self._memory.get(
                    MOTOR_TEMPERATURE.address,
                    ambient,
                )
            ),
            ambient,
        )

        self._memory[VIBRATION_RMS.address] = max(
            0.02,
            self._noise(0.02) + 0.05,
        )

        self._memory[CURRENT.address] = 0.0
        self._memory[POWER.address] = 0.0

    # --------------------------------------------------
    # Physical behavior
    # --------------------------------------------------

    def _calculate_load(self) -> float:
        base = (
            self.config.nominal_load_percent
        )

        variation = (
            8.0
            * math.sin(
                time.monotonic() / 20.0
            )
        )

        degradation_load = (
            self.state.degradation * 8.0
        )

        return self._clamp(
            base
            + variation
            + degradation_load
            + self._noise(2.0),
            10.0,
            100.0,
        )

    def _calculate_speed(
        self,
        load: float,
    ) -> float:
        speed = (
            self.config.nominal_speed_rpm
            - max(
                0.0,
                load
                - self.config.nominal_load_percent,
            )
            * 1.2
            + self._noise(8.0)
        )

        return self._clamp(
            speed,
            0.0,
            5000.0,
        )

    def _bearing_temperature(
        self,
        ambient: float,
        load: float,
    ) -> float:
        value = (
            ambient
            + 18.0
            + load * 0.30
            + self.state.degradation * 30.0
            + self._noise(1.0)
        )

        if (
            self.state.failure_mode
            == FailureMode.BEARING_WEAR
        ):
            value += 20.0

        return value

    def _motor_temperature(
        self,
        ambient: float,
        load: float,
    ) -> float:
        value = (
            ambient
            + 20.0
            + load * 0.35
            + self.state.degradation * 20.0
            + self._noise(1.2)
        )

        if (
            self.state.failure_mode
            == FailureMode.MOTOR_OVERHEAT
        ):
            value += 35.0

        return value

    def _vibration(
        self,
        load: float,
    ) -> float:
        value = (
            0.8
            + load * 0.018
            + self.state.degradation * 3.0
            + self._noise(0.12)
        )

        if (
            self.state.failure_mode
            == FailureMode.BEARING_WEAR
        ):
            value += 3.5

        if (
            self.state.failure_mode
            == FailureMode.VIBRATION
        ):
            value += 5.0

        return max(
            0.0,
            value,
        )

    def _oil_pressure(
        self,
        load: float,
    ) -> float:
        pressure = (
            4.5
            + load * 0.015
            - self.state.degradation * 0.8
            + self._noise(0.08)
        )

        if (
            self.state.failure_mode
            == FailureMode.OIL_PRESSURE
        ):
            pressure -= 3.5

        return max(
            0.0,
            pressure,
        )

    def _current(
        self,
        load: float,
    ) -> float:
        current = (
            12.0
            + load * 0.55
            + self.state.degradation * 8.0
            + self._noise(1.0)
        )

        if (
            self.state.failure_mode
            == FailureMode.OVERCURRENT
        ):
            current += 35.0

        return max(
            0.0,
            current,
        )

    # --------------------------------------------------
    # Production
    # --------------------------------------------------

    def _update_production(
        self,
        elapsed: float,
        load: float,
    ) -> None:
        expected_per_second = (
            0.20
            * load / 100.0
        )

        probability = min(
            1.0,
            expected_per_second * elapsed,
        )

        if self._rng.random() < probability:
            self.state.production_count += 1

            scrap_probability = (
                0.01
                + self.state.degradation * 0.08
            )

            if (
                self.state.failure_mode
                != FailureMode.NONE
            ):
                scrap_probability += 0.05

            if (
                self._rng.random()
                < scrap_probability
            ):
                self.state.scrap_count += 1

    # --------------------------------------------------
    # Failure simulation
    # --------------------------------------------------

    def inject_failure(
        self,
        failure_mode: FailureMode,
    ) -> None:
        self.state.failure_mode = failure_mode

        if failure_mode == FailureMode.NONE:
            if self.state.mode == MachineMode.DEGRADED:
                self.state.mode = MachineMode.RUNNING

        elif self.state.mode == MachineMode.RUNNING:
            self.state.mode = MachineMode.DEGRADED

        logger.warning(
            "Failure injected asset=%s failure=%s",
            self.config.asset_id,
            failure_mode.value,
        )

    def clear_failure(self) -> None:
        self.inject_failure(
            FailureMode.NONE
        )

    def set_degradation(
        self,
        degradation: float,
    ) -> None:
        self.state.degradation = self._clamp(
            float(degradation),
            0.0,
            1.0,
        )

    def accelerate_degradation(
        self,
        amount: float = 0.1,
    ) -> None:
        self.set_degradation(
            self.state.degradation + amount
        )

    # --------------------------------------------------
    # Emergency simulation
    # --------------------------------------------------

    async def simulate_fire(self) -> None:
        self.state.fire_detected = True
        self.state.smoke_detected = True

        self._memory[FIRE_DETECTED.address] = True
        self._memory[SMOKE_DETECTED.address] = True

        logger.warning(
            "Simulated fire asset=%s",
            self.config.asset_id,
        )

    async def simulate_smoke(self) -> None:
        self.state.smoke_detected = True

        self._memory[SMOKE_DETECTED.address] = True

    async def simulate_overheating(self) -> None:
        self.inject_failure(
            FailureMode.MOTOR_OVERHEAT
        )

        self._memory[
            MOTOR_TEMPERATURE.address
        ] = 125.0

        self._memory[OVERHEATING.address] = True

    # --------------------------------------------------
    # Safety
    # --------------------------------------------------

    def _evaluate_safety(self) -> None:
        motor_temp = float(
            self._memory[MOTOR_TEMPERATURE.address]
        )

        bearing_temp = float(
            self._memory[
                BEARING_TEMPERATURE.address
            ]
        )

        overheating = (
            motor_temp >= 105.0
            or bearing_temp >= 95.0
        )

        self._memory[
            OVERHEATING.address
        ] = overheating

        if overheating:
            logger.warning(
                "Overheating detected asset=%s "
                "motor=%.2f bearing=%.2f",
                self.config.asset_id,
                motor_temp,
                bearing_temp,
            )

        self._memory[
            FIRE_DETECTED.address
        ] = self.state.fire_detected

        self._memory[
            SMOKE_DETECTED.address
        ] = self.state.smoke_detected

        self._memory[
            EMERGENCY_STOP.address
        ] = self.state.emergency_stop

    # --------------------------------------------------
    # Predictive maintenance
    # --------------------------------------------------

    def _update_health(self) -> None:
        degradation = self.state.degradation

        vibration = float(
            self._memory.get(
                VIBRATION_RMS.address,
                0.0,
            )
        )

        motor_temp = float(
            self._memory.get(
                MOTOR_TEMPERATURE.address,
                self.config.ambient_temperature_c,
            )
        )

        bearing_temp = float(
            self._memory.get(
                BEARING_TEMPERATURE.address,
                self.config.ambient_temperature_c,
            )
        )

        oil_pressure = float(
            self._memory.get(
                OIL_PRESSURE.address,
                4.5,
            )
        )

        risk = degradation * 0.55

        if vibration >= 4.5:
            risk += 0.15

        if vibration >= 7.1:
            risk += 0.15

        if motor_temp >= 90:
            risk += 0.10

        if motor_temp >= 110:
            risk += 0.15

        if bearing_temp >= 80:
            risk += 0.10

        if oil_pressure <= 1.5:
            risk += 0.20

        if (
            self.state.failure_mode
            != FailureMode.NONE
        ):
            risk += 0.20

        failure_probability = self._clamp(
            risk,
            0.01,
            0.99,
        )

        health_score = self._clamp(
            100.0
            * (1.0 - failure_probability),
            0.0,
            100.0,
        )

        rul_hours = max(
            0.0,
            5000.0
            * (1.0 - degradation)
            * (1.0 - failure_probability * 0.65),
        )

        self._memory[
            FAILURE_PROBABILITY.address
        ] = round(
            failure_probability,
            5,
        )

        self._memory[
            HEALTH_SCORE.address
        ] = round(
            health_score,
            3,
        )

        self._memory[
            RUL_HOURS.address
        ] = round(
            rul_hours,
            3,
        )

    # --------------------------------------------------
    # Snapshots / streaming
    # --------------------------------------------------

    def get_snapshot(self) -> dict[str, Any]:
        return {
            "site_id": self.config.site_id,
            "asset_id": self.config.asset_id,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "machine_mode": self.state.mode.value,
            "failure_mode": (
                self.state.failure_mode.value
            ),
            "degradation": round(
                self.state.degradation,
                5,
            ),
            "telemetry": {
                tag.name: self._memory.get(
                    tag.address
                )
                for tag in SIMULATOR_TAGS
            },
        }

    async def stream(
        self,
        *,
        interval: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        delay = (
            interval
            if interval is not None
            else self.config.interval_seconds
        )

        if delay <= 0:
            raise ValueError(
                "Stream interval must be positive"
            )

        while self.is_connected:
            yield self.get_snapshot()

            await asyncio.sleep(delay)

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def _noise(
        self,
        magnitude: float,
    ) -> float:
        return self._rng.uniform(
            -magnitude,
            magnitude,
        )

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    @staticmethod
    def _cool_towards(
        value: float,
        target: float,
        rate: float = 0.08,
    ) -> float:
        return value + (
            target - value
        ) * rate

    async def __aenter__(
        self,
    ) -> "IndustrialPLCSimulator":
        await self.connect()
        await self.start_simulation()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        await self.stop_simulation()
        await self.disconnect()


def create_plc_simulator(
    *,
    asset_id: str = "MACHINE-001",
    site_id: str = "PLANT-001",
    interval_seconds: float = 1.0,
    random_seed: int | None = None,
) -> IndustrialPLCSimulator:
    config = SimulatorConfig(
        asset_id=asset_id,
        site_id=site_id,
        interval_seconds=interval_seconds,
        random_seed=random_seed,
    )

    return IndustrialPLCSimulator(config)