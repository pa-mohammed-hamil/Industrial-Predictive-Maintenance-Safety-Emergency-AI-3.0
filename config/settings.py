# app/config/settings.py

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration.

    Values are loaded from:
        1. Environment variables
        2. .env
        3. Defaults defined below

    Environment variables are case-insensitive.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =====================================================
    # Application
    # =====================================================

    app_name: str = (
        "Industrial Predictive Maintenance "
        "Safety & Emergency Intelligence AI"
    )

    app_version: str = "1.0.0"

    app_env: Literal[
        "development",
        "testing",
        "staging",
        "production",
    ] = "development"

    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8000

    api_prefix: str = "/api/v1"

    frontend_url: str = (
        "http://localhost:8000"
    )

    # =====================================================
    # CORS
    # =====================================================

    cors_origins: str = (
        "http://localhost:8000,"
        "http://127.0.0.1:8000,"
        "http://localhost:3000,"
        "http://localhost:5173"
    )

    cors_allow_credentials: bool = True

    @property
    def cors_origin_list(
        self,
    ) -> list[str]:
        return [
            value.strip()
            for value
            in self.cors_origins.split(",")
            if value.strip()
        ]

    # =====================================================
    # PostgreSQL
    # =====================================================

    database_url: str = (
        "postgresql+asyncpg://"
        "postgres:postgres@localhost:5432/"
        "industrial_ai"
    )

    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    sql_echo: bool = False

    # =====================================================
    # Redis
    # =====================================================

    redis_url: str = (
        "redis://localhost:6379/0"
    )

    redis_max_connections: int = 50
    redis_socket_timeout: float = 5.0
    redis_connect_timeout: float = 5.0

    redis_health_check_interval: int = 30

    redis_default_ttl: int = 300

    redis_key_prefix: str = (
        "industrial-ai"
    )

    # =====================================================
    # Kafka
    # =====================================================

    kafka_bootstrap_servers: str = (
        "localhost:9092"
    )

    kafka_client_id: str = (
        "industrial-ai"
    )

    kafka_consumer_group: str = (
        "industrial-ai-workers"
    )

    kafka_auto_offset_reset: Literal[
        "earliest",
        "latest",
    ] = "latest"

    kafka_enable_auto_commit: bool = True

    kafka_request_timeout_ms: int = 30000
    kafka_session_timeout_ms: int = 10000
    kafka_max_poll_records: int = 500

    kafka_security_protocol: str = (
        "PLAINTEXT"
    )

    kafka_sasl_mechanism: (
        str | None
    ) = None

    kafka_sasl_username: (
        str | None
    ) = None

    kafka_sasl_password: (
        SecretStr | None
    ) = None

    # =====================================================
    # MQTT
    # =====================================================

    mqtt_enabled: bool = True

    mqtt_host: str = "localhost"
    mqtt_port: int = 1883

    mqtt_client_id: str = (
        "industrial-ai-api"
    )

    mqtt_username: str | None = None

    mqtt_password: (
        SecretStr | None
    ) = None

    mqtt_keepalive: int = 60

    mqtt_qos: int = 1

    mqtt_use_tls: bool = False

    mqtt_topic_prefix: str = (
        "industrial"
    )

    # =====================================================
    # OPC-UA
    # =====================================================

    opcua_enabled: bool = False

    opcua_url: str = (
        "opc.tcp://localhost:4840"
    )

    opcua_username: str | None = None

    opcua_password: (
        SecretStr | None
    ) = None

    opcua_timeout: float = 10.0

    # =====================================================
    # PLC
    # =====================================================

    plc_mode: Literal[
        "simulator",
        "real",
    ] = "simulator"

    plc_poll_interval: float = 1.0

    # =====================================================
    # Authentication / JWT
    # =====================================================

    jwt_secret_key: SecretStr = Field(
        default=(
            "development-only-change-this-"
            "secret-key-before-production"
        )
    )

    jwt_algorithm: str = "HS256"

    jwt_issuer: str = "industrial-ai"

    jwt_audience: str = (
        "industrial-ai-api"
    )

    access_token_expire_minutes: int = 30

    refresh_token_expire_days: int = 7

    api_key_pepper: SecretStr = Field(
        default=(
            "development-api-key-pepper-"
            "change-in-production"
        )
    )

    # =====================================================
    # Rate limiting
    # =====================================================

    rate_limit_enabled: bool = True

    rate_limit_requests: int = 100

    rate_limit_window_seconds: int = 60

    # =====================================================
    # Logging
    # =====================================================

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    log_format: Literal[
        "json",
        "text",
    ] = "json"

    log_dir: str = "logs"

    log_file: str = (
        "industrial-ai.log"
    )

    error_log_file: str = (
        "error.log"
    )

    audit_log_file: str = (
        "audit.log"
    )

    security_log_file: str = (
        "security.log"
    )

    max_log_bytes: int = (
        20 * 1024 * 1024
    )

    log_backup_count: int = 10

    # =====================================================
    # Object storage
    # =====================================================

    storage_provider: Literal[
        "local",
        "s3",
        "minio",
    ] = "local"

    storage_path: str = "data/storage"

    aws_region: str = (
        "ap-south-1"
    )

    aws_access_key_id: (
        str | None
    ) = None

    aws_secret_access_key: (
        SecretStr | None
    ) = None

    s3_bucket: str = (
        "industrial-ai"
    )

    minio_endpoint: str = (
        "http://localhost:9000"
    )

    minio_access_key: (
        str | None
    ) = None

    minio_secret_key: (
        SecretStr | None
    ) = None

    # =====================================================
    # Vector database
    # =====================================================

    vector_db_provider: Literal[
        "qdrant",
        "faiss",
        "chroma",
    ] = "qdrant"

    qdrant_url: str = (
        "http://localhost:6333"
    )

    qdrant_api_key: (
        SecretStr | None
    ) = None

    qdrant_collection: str = (
        "industrial_documents"
    )

    # =====================================================
    # RAG
    # =====================================================

    rag_enabled: bool = True

    rag_top_k: int = 10

    rag_rerank_top_k: int = 5

    rag_chunk_size: int = 800

    rag_chunk_overlap: int = 120

    rag_hybrid_search: bool = True

    rag_reranking_enabled: bool = True

    rag_citations_enabled: bool = True

    embedding_model: str = (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )

    reranker_model: str = (
        "cross-encoder/"
        "ms-marco-MiniLM-L-6-v2"
    )

    # =====================================================
    # LLM
    # =====================================================

    llm_provider: str = "local"

    llm_model: str = (
        "default"
    )

    llm_api_key: (
        SecretStr | None
    ) = None

    llm_base_url: (
        str | None
    ) = None

    llm_temperature: float = 0.1

    llm_max_tokens: int = 2048

    # =====================================================
    # ML / predictive maintenance
    # =====================================================

    ml_model_dir: str = (
        "models"
    )

    model_registry_path: str = (
        "models/registry"
    )

    default_rul_model: str = (
        "rul_prediction"
    )

    default_failure_model: str = (
        "failure_prediction"
    )

    default_anomaly_model: str = (
        "multivariate_anomaly"
    )

    anomaly_threshold: float = 0.75

    failure_risk_threshold: float = 0.70

    critical_failure_threshold: float = 0.90

    model_cache_enabled: bool = True

    # =====================================================
    # Computer vision
    # =====================================================

    cv_enabled: bool = True

    cv_model_dir: str = (
        "models/cv"
    )

    cv_confidence_threshold: float = 0.50

    ppe_detection_enabled: bool = True
    fire_detection_enabled: bool = True
    smoke_detection_enabled: bool = True
    defect_detection_enabled: bool = True

    # =====================================================
    # Sensors
    # =====================================================

    sensor_simulation_enabled: bool = True

    telemetry_interval_seconds: float = 1.0

    temperature_warning_c: float = 75.0
    temperature_critical_c: float = 90.0

    vibration_warning_rms: float = 4.5
    vibration_critical_rms: float = 7.1

    pressure_warning_bar: float = 7.0
    pressure_critical_bar: float = 9.0

    current_warning_a: float = 80.0
    current_critical_a: float = 100.0

    # =====================================================
    # Safety
    # =====================================================

    safety_engine_enabled: bool = True

    emergency_engine_enabled: bool = True

    auto_escalation_enabled: bool = True

    emergency_shutdown_enabled: bool = False

    # =====================================================
    # Predictive maintenance
    # =====================================================

    predictive_maintenance_enabled: bool = True

    maintenance_prediction_hours: int = 168

    rul_warning_hours: float = 168.0

    rul_critical_hours: float = 48.0

    # =====================================================
    # Cost engine
    # =====================================================

    cost_engine_enabled: bool = True

    default_currency: str = "INR"

    electricity_cost_per_kwh: float = 8.0

    downtime_cost_per_hour: float = 5000.0

    # =====================================================
    # Email
    # =====================================================

    smtp_enabled: bool = False

    smtp_host: str = "localhost"
    smtp_port: int = 587

    smtp_username: str | None = None

    smtp_password: (
        SecretStr | None
    ) = None

    smtp_use_tls: bool = True

    email_from: str = (
        "industrial-ai@localhost"
    )

    # =====================================================
    # SMS
    # =====================================================

    sms_enabled: bool = False

    sms_provider_url: (
        str | None
    ) = None

    sms_api_key: (
        SecretStr | None
    ) = None

    # =====================================================
    # Webhook
    # =====================================================

    webhook_enabled: bool = True

    webhook_timeout_seconds: float = 10.0

    webhook_max_retries: int = 3

    webhook_secret: (
        SecretStr | None
    ) = None

    # =====================================================
    # WebSocket
    # =====================================================

    websocket_enabled: bool = True

    websocket_heartbeat_seconds: int = 30

    websocket_max_connections: int = 1000

    # =====================================================
    # Workers
    # =====================================================

    celery_enabled: bool = False

    celery_broker_url: str = (
        "redis://localhost:6379/1"
    )

    celery_result_backend: str = (
        "redis://localhost:6379/2"
    )

    # =====================================================
    # Observability
    # =====================================================

    metrics_enabled: bool = True

    prometheus_enabled: bool = True

    tracing_enabled: bool = False

    otel_service_name: str = (
        "industrial-ai"
    )

    otel_exporter_endpoint: (
        str | None
    ) = None

    # =====================================================
    # Feature flags
    # =====================================================

    enable_agents: bool = True
    enable_rag: bool = True
    enable_cv: bool = True
    enable_cost_engine: bool = True
    enable_safety_engine: bool = True
    enable_notifications: bool = True

    # =====================================================
    # Validation
    # =====================================================

    @field_validator("port")
    @classmethod
    def validate_port(
        cls,
        value: int,
    ) -> int:
        if not 1 <= value <= 65535:
            raise ValueError(
                "PORT must be between "
                "1 and 65535."
            )

        return value

    @field_validator(
        "llm_temperature"
    )
    @classmethod
    def validate_temperature(
        cls,
        value: float,
    ) -> float:
        if not 0 <= value <= 2:
            raise ValueError(
                "LLM temperature must "
                "be between 0 and 2."
            )

        return value

    @field_validator(
        "anomaly_threshold",
        "failure_risk_threshold",
        "critical_failure_threshold",
        "cv_confidence_threshold",
    )
    @classmethod
    def validate_probability(
        cls,
        value: float,
    ) -> float:
        if not 0 <= value <= 1:
            raise ValueError(
                "Probability threshold must "
                "be between 0 and 1."
            )

        return value

    # =====================================================
    # Convenience properties
    # =====================================================

    @property
    def is_development(self) -> bool:
        return (
            self.app_env
            == "development"
        )

    @property
    def is_testing(self) -> bool:
        return (
            self.app_env
            == "testing"
        )

    @property
    def is_production(self) -> bool:
        return (
            self.app_env
            == "production"
        )

    @property
    def jwt_secret(self) -> str:
        return (
            self.jwt_secret_key
            .get_secret_value()
        )

    @property
    def api_pepper(self) -> str:
        return (
            self.api_key_pepper
            .get_secret_value()
        )


# =========================================================
# Cached settings instance
# =========================================================

@lru_cache
def get_settings() -> Settings:
    """
    Return one cached Settings instance.

    FastAPI:
        settings = Depends(get_settings)
    """

    return Settings()


settings = get_settings()