# app/integrations/opcua/client.py

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Sequence

try:
    from asyncua import Client, Node, ua
except ImportError as exc:
    raise ImportError(
        "asyncua is required for OPC-UA support. "
        "Install it with: pip install asyncua"
    ) from exc


logger = logging.getLogger(__name__)

ValueCallback = Callable[
    [str, Any, datetime],
    Awaitable[None] | None,
]


@dataclass(slots=True)
class OPCUAConfig:
    endpoint: str = "opc.tcp://localhost:4840"

    username: str | None = None
    password: str | None = None

    timeout: float = 10.0
    session_timeout: float = 60_000.0

    security_policy: str | None = None
    certificate_path: str | None = None
    private_key_path: str | None = None

    subscription_interval_ms: float = 1000.0

    reconnect_enabled: bool = True
    reconnect_delay: float = 2.0
    reconnect_max_delay: float = 30.0


@dataclass(slots=True)
class OPCUAValue:
    node_id: str
    value: Any
    timestamp: datetime
    status: str = "Good"


@dataclass(slots=True)
class OPCUANodeDefinition:
    node_id: str
    name: str | None = None
    unit: str | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class SubscriptionHandler:
    """Receives OPC-UA data-change notifications."""

    def __init__(
        self,
        callback: ValueCallback,
    ) -> None:
        self.callback = callback

    def datachange_notification(
        self,
        node: Node,
        value: Any,
        data: Any,
    ) -> None:
        try:
            node_id = node.nodeid.to_string()

            timestamp = datetime.now(timezone.utc)

            monitored_item = getattr(
                data,
                "monitored_item",
                None,
            )

            if monitored_item is not None:
                server_timestamp = getattr(
                    monitored_item.Value,
                    "ServerTimestamp",
                    None,
                )

                if server_timestamp:
                    timestamp = server_timestamp

            result = self.callback(
                node_id,
                value,
                timestamp,
            )

            if asyncio.iscoroutine(result):
                asyncio.create_task(result)

        except Exception:
            logger.exception(
                "OPC-UA subscription callback failed"
            )

    def event_notification(
        self,
        event: Any,
    ) -> None:
        logger.debug(
            "OPC-UA event received: %s",
            event,
        )


class OPCUAClient:
    """
    Async OPC-UA client for industrial equipment.

    Supports:
    - authenticated connections
    - secure OPC-UA endpoints
    - node reads/writes
    - bulk reads
    - subscriptions
    - telemetry callbacks
    - reconnect
    - health checks
    - context-manager lifecycle
    """

    def __init__(
        self,
        config: OPCUAConfig,
    ) -> None:
        self.config = config

        self._client: Client | None = None

        self._connected = False

        self._subscriptions: list[Any] = []

        self._subscription_handlers: list[
            SubscriptionHandler
        ] = []

        self._lock = asyncio.Lock()

    # --------------------------------------------------
    # Connection
    # --------------------------------------------------

    async def connect(self) -> None:
        async with self._lock:

            if self._connected:
                return

            logger.info(
                "Connecting to OPC-UA server %s",
                self.config.endpoint,
            )

            client = Client(
                url=self.config.endpoint,
                timeout=self.config.timeout,
            )

            client.session_timeout = (
                self.config.session_timeout
            )

            if self.config.username:
                client.set_user(
                    self.config.username
                )

            if self.config.password:
                client.set_password(
                    self.config.password
                )

            if self.config.security_policy:
                await self._configure_security(
                    client
                )

            try:
                await client.connect()

            except Exception:
                logger.exception(
                    "Failed connecting to OPC-UA server"
                )
                raise

            self._client = client
            self._connected = True

            logger.info(
                "Connected to OPC-UA server %s",
                self.config.endpoint,
            )

    async def disconnect(self) -> None:
        async with self._lock:

            if not self._client:
                return

            logger.info(
                "Disconnecting OPC-UA client"
            )

            for subscription in self._subscriptions:
                try:
                    await subscription.delete()
                except Exception:
                    logger.exception(
                        "Failed deleting OPC-UA subscription"
                    )

            self._subscriptions.clear()
            self._subscription_handlers.clear()

            try:
                await self._client.disconnect()

            finally:
                self._connected = False
                self._client = None

            logger.info(
                "OPC-UA client disconnected"
            )

    async def _configure_security(
        self,
        client: Client,
    ) -> None:

        if not self.config.certificate_path:
            raise ValueError(
                "certificate_path required when "
                "OPC-UA security is enabled"
            )

        if not self.config.private_key_path:
            raise ValueError(
                "private_key_path required when "
                "OPC-UA security is enabled"
            )

        security = (
            f"{self.config.security_policy},"
            f"SignAndEncrypt,"
            f"{self.config.certificate_path},"
            f"{self.config.private_key_path}"
        )

        await client.set_security_string(
            security
        )

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _require_client(self) -> Client:
        if not self._connected:
            raise ConnectionError(
                "OPC-UA client is not connected"
            )

        if self._client is None:
            raise ConnectionError(
                "OPC-UA client unavailable"
            )

        return self._client

    def get_node(
        self,
        node_id: str,
    ) -> Node:

        if not node_id:
            raise ValueError(
                "OPC-UA node_id cannot be empty"
            )

        client = self._require_client()

        return client.get_node(node_id)

    # --------------------------------------------------
    # Read
    # --------------------------------------------------

    async def read_value(
        self,
        node_id: str,
    ) -> Any:

        node = self.get_node(node_id)

        try:
            return await node.read_value()

        except Exception:
            logger.exception(
                "Failed reading OPC-UA node %s",
                node_id,
            )
            raise

    async def read_data_value(
        self,
        node_id: str,
    ) -> OPCUAValue:

        node = self.get_node(node_id)

        try:
            data_value = (
                await node.read_data_value()
            )

            timestamp = (
                data_value.ServerTimestamp
                or data_value.SourceTimestamp
                or datetime.now(timezone.utc)
            )

            status = str(
                data_value.StatusCode
            )

            return OPCUAValue(
                node_id=node_id,
                value=data_value.Value.Value,
                timestamp=timestamp,
                status=status,
            )

        except Exception:
            logger.exception(
                "Failed reading OPC-UA data value %s",
                node_id,
            )
            raise

    async def read_values(
        self,
        node_ids: Sequence[str],
    ) -> dict[str, Any]:

        if not node_ids:
            return {}

        client = self._require_client()

        nodes = [
            client.get_node(node_id)
            for node_id in node_ids
        ]

        try:
            values = await client.read_values(
                nodes
            )

        except Exception:
            logger.exception(
                "OPC-UA bulk read failed"
            )
            raise

        return dict(
            zip(
                node_ids,
                values,
                strict=True,
            )
        )

    async def read_telemetry(
        self,
        nodes: Sequence[
            OPCUANodeDefinition
        ],
    ) -> dict[str, OPCUAValue]:

        results: dict[str, OPCUAValue] = {}

        for definition in nodes:
            try:
                value = await self.read_data_value(
                    definition.node_id
                )

                key = (
                    definition.name
                    or definition.node_id
                )

                results[key] = value

            except Exception:
                logger.exception(
                    "Telemetry read failed node=%s",
                    definition.node_id,
                )

        return results

    # --------------------------------------------------
    # Write
    # --------------------------------------------------

    async def write_value(
        self,
        node_id: str,
        value: Any,
    ) -> None:

        node = self.get_node(node_id)

        try:
            await node.write_value(value)

            logger.debug(
                "OPC-UA write node=%s value=%s",
                node_id,
                value,
            )

        except Exception:
            logger.exception(
                "Failed writing OPC-UA node %s",
                node_id,
            )
            raise

    async def write_typed_value(
        self,
        node_id: str,
        value: Any,
        variant_type: ua.VariantType,
    ) -> None:

        node = self.get_node(node_id)

        variant = ua.Variant(
            value,
            variant_type,
        )

        try:
            await node.write_value(variant)

        except Exception:
            logger.exception(
                "Typed OPC-UA write failed node=%s",
                node_id,
            )
            raise

    # --------------------------------------------------
    # Browse
    # --------------------------------------------------

    async def browse_children(
        self,
        node_id: str,
    ) -> list[dict[str, str]]:

        node = self.get_node(node_id)

        children = await node.get_children()

        result: list[dict[str, str]] = []

        for child in children:

            browse_name = (
                await child.read_browse_name()
            )

            display_name = (
                await child.read_display_name()
            )

            result.append(
                {
                    "node_id": (
                        child.nodeid.to_string()
                    ),
                    "browse_name": (
                        browse_name.Name
                    ),
                    "display_name": (
                        display_name.Text
                    ),
                }
            )

        return result

    # --------------------------------------------------
    # Subscriptions
    # --------------------------------------------------

    async def subscribe(
        self,
        node_ids: Sequence[str],
        callback: ValueCallback,
        *,
        interval_ms: float | None = None,
    ) -> Any:

        if not node_ids:
            raise ValueError(
                "At least one OPC-UA node required"
            )

        client = self._require_client()

        handler = SubscriptionHandler(
            callback
        )

        interval = (
            interval_ms
            if interval_ms is not None
            else self.config.subscription_interval_ms
        )

        subscription = (
            await client.create_subscription(
                interval,
                handler,
            )
        )

        nodes = [
            client.get_node(node_id)
            for node_id in node_ids
        ]

        try:
            handles = (
                await subscription.subscribe_data_change(
                    nodes
                )
            )

        except Exception:
            await subscription.delete()
            raise

        self._subscriptions.append(
            subscription
        )

        self._subscription_handlers.append(
            handler
        )

        logger.info(
            "Subscribed to %s OPC-UA nodes",
            len(node_ids),
        )

        return handles

    # --------------------------------------------------
    # Method invocation
    # --------------------------------------------------

    async def call_method(
        self,
        object_node_id: str,
        method_node_id: str,
        *args: Any,
    ) -> Any:

        object_node = self.get_node(
            object_node_id
        )

        method_node = self.get_node(
            method_node_id
        )

        try:
            return await object_node.call_method(
                method_node,
                *args,
            )

        except Exception:
            logger.exception(
                "OPC-UA method call failed "
                "object=%s method=%s",
                object_node_id,
                method_node_id,
            )
            raise

    # --------------------------------------------------
    # Health
    # --------------------------------------------------

    async def health_check(self) -> bool:

        if not self._connected:
            return False

        try:
            client = self._require_client()

            server_node = client.get_node(
                "i=2253"
            )

            await asyncio.wait_for(
                server_node.read_value(),
                timeout=self.config.timeout,
            )

            return True

        except Exception:
            logger.warning(
                "OPC-UA health check failed"
            )

            return False

    # --------------------------------------------------
    # Reconnection
    # --------------------------------------------------

    async def reconnect(self) -> bool:

        if not self.config.reconnect_enabled:
            return False

        delay = max(
            self.config.reconnect_delay,
            0.1,
        )

        maximum = max(
            self.config.reconnect_max_delay,
            delay,
        )

        while True:

            try:
                await self.disconnect()

            except Exception:
                pass

            try:
                await self.connect()

                logger.info(
                    "OPC-UA reconnect successful"
                )

                return True

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.warning(
                    "OPC-UA reconnect failed; "
                    "retrying in %.1fs",
                    delay,
                )

            await asyncio.sleep(delay)

            delay = min(
                delay * 2,
                maximum,
            )

    # --------------------------------------------------
    # Properties / context manager
    # --------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def endpoint(self) -> str:
        return self.config.endpoint

    async def __aenter__(
        self,
    ) -> "OPCUAClient":

        await self.connect()

        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:

        await self.disconnect()


def create_opcua_client(
    *,
    endpoint: str,
    username: str | None = None,
    password: str | None = None,
    timeout: float = 10.0,
    subscription_interval_ms: float = 1000.0,
    security_policy: str | None = None,
    certificate_path: str | None = None,
    private_key_path: str | None = None,
) -> OPCUAClient:
    """Create the application's OPC-UA client."""

    config = OPCUAConfig(
        endpoint=endpoint,
        username=username,
        password=password,
        timeout=timeout,
        subscription_interval_ms=(
            subscription_interval_ms
        ),
        security_policy=security_policy,
        certificate_path=certificate_path,
        private_key_path=private_key_path,
    )

    return OPCUAClient(config)