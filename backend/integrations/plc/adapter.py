# app/integrations/plc/adapter.py

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


class PLCState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DEGRADED = "degraded"
    ERROR = "error"


class PLCDataType(str, Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"


class PLCAccess(str, Enum):
    READ_ONLY = "read_only"
    WRITE_ONLY = "write_only"
    READ_WRITE = "read_write"


@dataclass(frozen=True, slots=True)
class PLCTag:
    name: str
    address: str

    data_type: PLCDataType = PLCDataType.FLOAT
    access: PLCAccess = PLCAccess.READ_ONLY

    unit: str | None = None
    description: str | None = None

    minimum: float | None = None
    maximum: float | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    @property
    def readable(self) -> bool:
        return self.access in {
            PLCAccess.READ_ONLY,
            PLCAccess.READ_WRITE,
        }

    @property
    def writable(self) -> bool:
        return self.access in {
            PLCAccess.WRITE_ONLY,
            PLCAccess.READ_WRITE,
        }


@dataclass(slots=True)
class PLCValue:
    tag: str
    address: str
    value: Any

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    quality: str = "good"


@dataclass(slots=True)
class PLCHealth:
    state: PLCState
    connected: bool

    latency_ms: float | None = None
    error: str | None = None

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )


class PLCAdapter(ABC):
    """
    Base adapter for industrial PLC integrations.

    Implementations may target:
    - Siemens S7
    - Allen-Bradley / EtherNet/IP
    - Modbus TCP
    - OPC-UA-backed PLCs
    - simulated PLCs
    """

    def __init__(
        self,
        *,
        name: str,
    ) -> None:
        self.name = name

        self._state = PLCState.DISCONNECTED
        self._lock = asyncio.Lock()

    # --------------------------------------------------
    # Required driver methods
    # --------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the PLC."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the PLC."""

    @abstractmethod
    async def read_raw(
        self,
        address: str,
    ) -> Any:
        """Read a raw value from a PLC address."""

    @abstractmethod
    async def write_raw(
        self,
        address: str,
        value: Any,
    ) -> None:
        """Write a raw value to a PLC address."""

    # --------------------------------------------------
    # State
    # --------------------------------------------------

    @property
    def state(self) -> PLCState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == PLCState.CONNECTED

    def _set_state(
        self,
        state: PLCState,
    ) -> None:
        if state != self._state:
            logger.info(
                "PLC %s state %s -> %s",
                self.name,
                self._state.value,
                state.value,
            )

        self._state = state

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise ConnectionError(
                f"PLC {self.name!r} is not connected"
            )

    # --------------------------------------------------
    # Validated reads
    # --------------------------------------------------

    async def read(
        self,
        tag: PLCTag,
    ) -> PLCValue:
        self._require_connected()

        if not tag.readable:
            raise PermissionError(
                f"PLC tag {tag.name!r} is not readable"
            )

        try:
            raw = await self.read_raw(
                tag.address
            )

            value = self._convert_value(
                raw,
                tag.data_type,
            )

            return PLCValue(
                tag=tag.name,
                address=tag.address,
                value=value,
            )

        except Exception:
            logger.exception(
                "PLC read failed plc=%s tag=%s",
                self.name,
                tag.name,
            )
            raise

    async def read_many(
        self,
        tags: Sequence[PLCTag],
    ) -> dict[str, PLCValue]:
        if not tags:
            return {}

        results = await asyncio.gather(
            *(
                self.read(tag)
                for tag in tags
            ),
            return_exceptions=True,
        )

        values: dict[str, PLCValue] = {}

        for tag, result in zip(
            tags,
            results,
            strict=True,
        ):
            if isinstance(result, BaseException):
                logger.error(
                    "PLC bulk read failed "
                    "plc=%s tag=%s error=%s",
                    self.name,
                    tag.name,
                    result,
                )
                continue

            values[tag.name] = result

        return values

    # --------------------------------------------------
    # Validated writes
    # --------------------------------------------------

    async def write(
        self,
        tag: PLCTag,
        value: Any,
    ) -> None:
        self._require_connected()

        if not tag.writable:
            raise PermissionError(
                f"PLC tag {tag.name!r} is not writable"
            )

        converted = self._convert_value(
            value,
            tag.data_type,
        )

        self._validate_range(
            tag,
            converted,
        )

        try:
            await self.write_raw(
                tag.address,
                converted,
            )

            logger.info(
                "PLC write plc=%s tag=%s",
                self.name,
                tag.name,
            )

        except Exception:
            logger.exception(
                "PLC write failed plc=%s tag=%s",
                self.name,
                tag.name,
            )
            raise

    async def write_many(
        self,
        values: Sequence[
            tuple[PLCTag, Any]
        ],
    ) -> None:
        for tag, value in values:
            await self.write(
                tag,
                value,
            )

    # --------------------------------------------------
    # Telemetry
    # --------------------------------------------------

    async def collect_telemetry(
        self,
        tags: Sequence[PLCTag],
    ) -> dict[str, Any]:
        values = await self.read_many(tags)

        return {
            name: plc_value.value
            for name, plc_value in values.items()
        }

    async def telemetry_snapshot(
        self,
        tags: Sequence[PLCTag],
    ) -> dict[str, Any]:
        values = await self.read_many(tags)

        return {
            "plc": self.name,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "state": self.state.value,
            "telemetry": {
                name: {
                    "value": item.value,
                    "address": item.address,
                    "quality": item.quality,
                    "timestamp": (
                        item.timestamp.isoformat()
                    ),
                }
                for name, item in values.items()
            },
        }

    # --------------------------------------------------
    # Polling
    # --------------------------------------------------

    async def poll(
        self,
        tags: Sequence[PLCTag],
        *,
        interval: float = 1.0,
    ):
        """
        Async telemetry stream.

        Example:

            async for snapshot in adapter.poll(tags):
                await process(snapshot)
        """

        if interval <= 0:
            raise ValueError(
                "Polling interval must be positive"
            )

        while self.is_connected:
            yield await self.telemetry_snapshot(
                tags
            )

            await asyncio.sleep(interval)

    # --------------------------------------------------
    # Health
    # --------------------------------------------------

    async def health_check(
        self,
    ) -> PLCHealth:
        if not self.is_connected:
            return PLCHealth(
                state=self.state,
                connected=False,
            )

        loop = asyncio.get_running_loop()
        started = loop.time()

        try:
            await self._health_probe()

            latency = (
                loop.time() - started
            ) * 1000.0

            return PLCHealth(
                state=PLCState.CONNECTED,
                connected=True,
                latency_ms=latency,
            )

        except Exception as exc:
            logger.warning(
                "PLC health check failed "
                "plc=%s error=%s",
                self.name,
                exc,
            )

            return PLCHealth(
                state=PLCState.DEGRADED,
                connected=False,
                error=str(exc),
            )

    async def _health_probe(self) -> None:
        """
        Override in drivers when a protocol-specific
        health check is available.
        """

        if not self.is_connected:
            raise ConnectionError(
                "PLC is disconnected"
            )

    # --------------------------------------------------
    # Reconnection
    # --------------------------------------------------

    async def reconnect(
        self,
        *,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        max_attempts: int | None = None,
    ) -> bool:
        delay = max(
            0.1,
            initial_delay,
        )

        maximum = max(
            delay,
            max_delay,
        )

        attempt = 0

        while (
            max_attempts is None
            or attempt < max_attempts
        ):
            attempt += 1

            try:
                await self.disconnect()
            except Exception:
                logger.debug(
                    "PLC disconnect during reconnect "
                    "failed",
                    exc_info=True,
                )

            try:
                self._set_state(
                    PLCState.CONNECTING
                )

                await self.connect()

                if self.is_connected:
                    logger.info(
                        "PLC reconnect successful "
                        "plc=%s attempt=%s",
                        self.name,
                        attempt,
                    )

                    return True

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                self._set_state(
                    PLCState.ERROR
                )

                logger.warning(
                    "PLC reconnect failed "
                    "plc=%s attempt=%s error=%s",
                    self.name,
                    attempt,
                    exc,
                )

            await asyncio.sleep(delay)

            delay = min(
                delay * 2,
                maximum,
            )

        return False

    # --------------------------------------------------
    # Value conversion
    # --------------------------------------------------

    @staticmethod
    def _convert_value(
        value: Any,
        data_type: PLCDataType,
    ) -> Any:
        if data_type == PLCDataType.BOOL:
            if isinstance(value, str):
                normalized = value.strip().lower()

                if normalized in {
                    "true",
                    "1",
                    "on",
                    "yes",
                }:
                    return True

                if normalized in {
                    "false",
                    "0",
                    "off",
                    "no",
                }:
                    return False

                raise ValueError(
                    f"Cannot convert {value!r} to bool"
                )

            return bool(value)

        if data_type == PLCDataType.INT:
            return int(value)

        if data_type == PLCDataType.FLOAT:
            return float(value)

        if data_type == PLCDataType.STRING:
            return str(value)

        raise ValueError(
            f"Unsupported PLC data type: {data_type}"
        )

    @staticmethod
    def _validate_range(
        tag: PLCTag,
        value: Any,
    ) -> None:
        if not isinstance(
            value,
            (int, float),
        ):
            return

        numeric = float(value)

        if (
            tag.minimum is not None
            and numeric < tag.minimum
        ):
            raise ValueError(
                f"{tag.name} value {numeric} "
                f"is below minimum {tag.minimum}"
            )

        if (
            tag.maximum is not None
            and numeric > tag.maximum
        ):
            raise ValueError(
                f"{tag.name} value {numeric} "
                f"is above maximum {tag.maximum}"
            )

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    async def __aenter__(
        self,
    ) -> "PLCAdapter":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        await self.disconnect()


class MemoryPLCAdapter(PLCAdapter):
    """
    In-memory PLC implementation.

    Useful for:
    - local development
    - unit tests
    - demo environments
    - CI pipelines
    """

    def __init__(
        self,
        *,
        name: str = "simulated-plc",
        initial_values: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> None:
        super().__init__(name=name)

        self._memory: dict[str, Any] = dict(
            initial_values or {}
        )

    async def connect(self) -> None:
        async with self._lock:
            self._set_state(
                PLCState.CONNECTING
            )

            await asyncio.sleep(0)

            self._set_state(
                PLCState.CONNECTED
            )

            logger.info(
                "Memory PLC connected plc=%s",
                self.name,
            )

    async def disconnect(self) -> None:
        async with self._lock:
            self._set_state(
                PLCState.DISCONNECTED
            )

    async def read_raw(
        self,
        address: str,
    ) -> Any:
        self._require_connected()

        if address not in self._memory:
            raise KeyError(
                f"PLC address not found: {address}"
            )

        return self._memory[address]

    async def write_raw(
        self,
        address: str,
        value: Any,
    ) -> None:
        self._require_connected()

        self._memory[address] = value

    def seed(
        self,
        values: Mapping[str, Any],
    ) -> None:
        self._memory.update(values)

    def snapshot(
        self,
    ) -> dict[str, Any]:
        return dict(self._memory)