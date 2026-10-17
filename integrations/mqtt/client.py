# app/integrations/mqtt/client.py

from __future__ import annotations

import asyncio
import json
import logging
import ssl
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

MessageHandler = Callable[
    [str, bytes],
    Awaitable[None] | None,
]


@dataclass(slots=True)
class MQTTConfig:
    host: str = "localhost"
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None

    client_id: str = "industrial-ai-api"

    keepalive: int = 60
    clean_session: bool = True

    qos: int = 1

    use_tls: bool = False
    ca_cert: Optional[str] = None
    client_cert: Optional[str] = None
    client_key: Optional[str] = None

    reconnect_min_delay: int = 1
    reconnect_max_delay: int = 30


class MQTTClient:
    """
    Async-friendly MQTT integration client.

    Supports:
    - MQTT connection lifecycle
    - TLS
    - Authentication
    - Automatic reconnect
    - Topic subscriptions
    - JSON publishing
    - Async message handlers
    """

    def __init__(
        self,
        config: MQTTConfig,
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.config = config
        self.loop = loop

        self._connected = asyncio.Event()
        self._stopping = False

        self._handlers: dict[str, list[MessageHandler]] = {}

        self._client = mqtt.Client(
            client_id=config.client_id,
            clean_session=config.clean_session,
            protocol=mqtt.MQTTv311,
        )

        self._configure_authentication()
        self._configure_tls()

        self._client.reconnect_delay_set(
            min_delay=config.reconnect_min_delay,
            max_delay=config.reconnect_max_delay,
        )

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    def _configure_authentication(self) -> None:
        if self.config.username:
            self._client.username_pw_set(
                username=self.config.username,
                password=self.config.password,
            )

    def _configure_tls(self) -> None:
        if not self.config.use_tls:
            return

        context = ssl.create_default_context(
            cafile=self.config.ca_cert,
        )

        if self.config.client_cert:
            context.load_cert_chain(
                certfile=self.config.client_cert,
                keyfile=self.config.client_key,
            )

        self._client.tls_set_context(context)

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    async def connect(self, timeout: float = 10.0) -> None:
        if self.loop is None:
            self.loop = asyncio.get_running_loop()

        self._stopping = False
        self._connected.clear()

        logger.info(
            "Connecting to MQTT broker %s:%s",
            self.config.host,
            self.config.port,
        )

        result = self._client.connect_async(
            host=self.config.host,
            port=self.config.port,
            keepalive=self.config.keepalive,
        )

        if result != mqtt.MQTT_ERR_SUCCESS:
            raise ConnectionError(
                f"MQTT connect initialization failed: {result}"
            )

        self._client.loop_start()

        try:
            await asyncio.wait_for(
                self._connected.wait(),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            self._client.loop_stop()

            raise TimeoutError(
                f"MQTT connection timeout: "
                f"{self.config.host}:{self.config.port}"
            ) from exc

    async def disconnect(self) -> None:
        self._stopping = True

        logger.info("Disconnecting MQTT client")

        try:
            self._client.disconnect()
        finally:
            self._client.loop_stop()
            self._connected.clear()

    async def wait_until_connected(
        self,
        timeout: Optional[float] = None,
    ) -> bool:
        try:
            if timeout is None:
                await self._connected.wait()
            else:
                await asyncio.wait_for(
                    self._connected.wait(),
                    timeout=timeout,
                )

            return True

        except asyncio.TimeoutError:
            return False

    # ---------------------------------------------------------
    # Subscription
    # ---------------------------------------------------------

    async def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        qos: Optional[int] = None,
    ) -> None:
        if not topic:
            raise ValueError("MQTT topic cannot be empty")

        qos = self.config.qos if qos is None else qos

        self._handlers.setdefault(topic, []).append(handler)

        if self._connected.is_set():
            result, _ = self._client.subscribe(
                topic,
                qos=qos,
            )

            if result != mqtt.MQTT_ERR_SUCCESS:
                raise RuntimeError(
                    f"Failed subscribing to {topic}: {result}"
                )

        logger.info(
            "Registered MQTT subscription topic=%s qos=%s",
            topic,
            qos,
        )

    async def unsubscribe(
        self,
        topic: str,
    ) -> None:
        self._handlers.pop(topic, None)

        if self._connected.is_set():
            result, _ = self._client.unsubscribe(topic)

            if result != mqtt.MQTT_ERR_SUCCESS:
                raise RuntimeError(
                    f"Failed unsubscribing from {topic}: {result}"
                )

    # ---------------------------------------------------------
    # Publishing
    # ---------------------------------------------------------

    async def publish(
        self,
        topic: str,
        payload: str | bytes,
        *,
        qos: Optional[int] = None,
        retain: bool = False,
    ) -> None:
        if not self._connected.is_set():
            raise ConnectionError(
                "MQTT client is not connected"
            )

        qos = self.config.qos if qos is None else qos

        info = self._client.publish(
            topic=topic,
            payload=payload,
            qos=qos,
            retain=retain,
        )

        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(
                f"MQTT publish failed: {info.rc}"
            )

        await asyncio.to_thread(info.wait_for_publish)

    async def publish_json(
        self,
        topic: str,
        data: dict[str, Any],
        *,
        qos: Optional[int] = None,
        retain: bool = False,
    ) -> None:
        payload = json.dumps(
            data,
            separators=(",", ":"),
            default=str,
        )

        await self.publish(
            topic=topic,
            payload=payload,
            qos=qos,
            retain=retain,
        )

    # ---------------------------------------------------------
    # MQTT callbacks
    # ---------------------------------------------------------

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: dict[str, Any],
        rc: int,
    ) -> None:
        if rc != 0:
            logger.error(
                "MQTT connection failed rc=%s",
                rc,
            )
            return

        logger.info(
            "Connected to MQTT broker %s:%s",
            self.config.host,
            self.config.port,
        )

        if self.loop:
            self.loop.call_soon_threadsafe(
                self._connected.set
            )

        self._restore_subscriptions()

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        rc: int,
    ) -> None:
        if self.loop:
            self.loop.call_soon_threadsafe(
                self._connected.clear
            )

        if self._stopping:
            logger.info("MQTT client disconnected")
        else:
            logger.warning(
                "Unexpected MQTT disconnect rc=%s. "
                "Automatic reconnect enabled.",
                rc,
            )

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        message: mqtt.MQTTMessage,
    ) -> None:
        if not self.loop:
            logger.error(
                "Cannot process MQTT message: "
                "asyncio loop unavailable"
            )
            return

        topic = message.topic
        payload = bytes(message.payload)

        handlers = self._matching_handlers(topic)

        for handler in handlers:
            asyncio.run_coroutine_threadsafe(
                self._execute_handler(
                    handler,
                    topic,
                    payload,
                ),
                self.loop,
            )

    # ---------------------------------------------------------
    # Handler execution
    # ---------------------------------------------------------

    async def _execute_handler(
        self,
        handler: MessageHandler,
        topic: str,
        payload: bytes,
    ) -> None:
        try:
            result = handler(topic, payload)

            if asyncio.iscoroutine(result):
                await result

        except Exception:
            logger.exception(
                "MQTT handler failed topic=%s",
                topic,
            )

    # ---------------------------------------------------------
    # Topic matching
    # ---------------------------------------------------------

    def _matching_handlers(
        self,
        topic: str,
    ) -> list[MessageHandler]:

        handlers: list[MessageHandler] = []

        for subscription, callbacks in self._handlers.items():
            if mqtt.topic_matches_sub(
                subscription,
                topic,
            ):
                handlers.extend(callbacks)

        return handlers

    # ---------------------------------------------------------
    # Reconnect subscriptions
    # ---------------------------------------------------------

    def _restore_subscriptions(self) -> None:
        for topic in self._handlers:
            result, _ = self._client.subscribe(
                topic,
                qos=self.config.qos,
            )

            if result != mqtt.MQTT_ERR_SUCCESS:
                logger.error(
                    "Failed restoring MQTT "
                    "subscription topic=%s rc=%s",
                    topic,
                    result,
                )

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    async def __aenter__(self) -> "MQTTClient":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        await self.disconnect()


def decode_json_payload(
    payload: bytes,
) -> dict[str, Any]:
    """
    Decode an MQTT message containing a JSON object.
    """

    try:
        decoded = payload.decode("utf-8")
        data = json.loads(decoded)

    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Invalid MQTT JSON payload"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "MQTT JSON payload must contain an object"
        )

    return data