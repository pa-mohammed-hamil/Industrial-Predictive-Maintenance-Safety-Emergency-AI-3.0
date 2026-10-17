
"""
app/core/env.py

Centralized environment configuration for Industrial AI.

Python 3.11+
Pydantic Settings 2.x

Install:
    pip install pydantic-settings python-dotenv
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    # ========================================================
    # 1. Application
    # ========================================================

    APP_NAME: str = "Industrial Predictive Maintenance AI"
    APP_ENV: Literal["development", "dev", "testing", "staging", "production"] = "development"
    DEBUG: bool = False

    API_HOST: str = "0.0.0.0"
    API_PORT: int = Field(default=8000, ge=1, le=65535)
    API_PREFIX: str = "/api/v1"

    LOG_LEVEL: Literal[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO"

    # ========================================================
    # 2. PostgreSQL
    # ========================================================

    DATABASE_URL: str | None = None

    DB_HOST: str = "localhost"
    DB_PORT: int = Field(default=5432, ge=1, le=65535)
    DB_NAME: str = "industrial_ai"
    DB_USER: str = "industrial_user"
    DB_PASSWORD: SecretStr = SecretStr("")

    DB_SSL_MODE: Literal[
        "disable", "allow", "prefer", "require",
        "verify-ca", "verify-full"
    ] = "prefer"

    DB_POOL_SIZE: int = Field(default=10, ge=1)
    DB_MAX_OVERFLOW: int = Field(default=20, ge=0)
    DB_POOL_TIMEOUT: int = Field(default=30, ge=1)

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL

        username = quote_plus(self.DB_USER)
        password = quote_plus(
            self.DB_PASSWORD.get_secret_value()
        )

        return (
            f"postgresql+psycopg://{username}:{password}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            f"?sslmode={self.DB_SSL_MODE}"
        )

    # ========================================================
    # 3. Redis
    # ========================================================

    REDIS_URL: str | None = None
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = Field(default=6379, ge=1, le=65535)
    REDIS_DB: int = Field(default=0, ge=0)
    REDIS_PASSWORD: SecretStr = SecretStr("")

    @property
    def redis_connection_url(self) -> str:
        if self.REDIS_URL:
            return self.REDIS_URL

        password = self.REDIS_PASSWORD.get_secret_value()
        auth = f":{quote_plus(password)}@" if password else ""

        return (
            f"redis://{auth}{self.REDIS_HOST}:"
            f"{self.REDIS_PORT}/{self.REDIS_DB}"
        )

    # ========================================================
    # 4. Celery
    # ========================================================

    CELERY_BROKER_URL: str | None = None
    CELERY_RESULT_BACKEND: str | None = None
    CELERY_WORKER_CONCURRENCY: int = Field(default=2, ge=1)

    @property
    def celery_broker(self) -> str:
        return self.CELERY_BROKER_URL or self.redis_connection_url

    @property
    def celery_backend(self) -> str:
        return (
            self.CELERY_RESULT_BACKEND
            or self.redis_connection_url
        )

    # ========================================================
    # 5. Kafka
    # ========================================================

    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_CONSUMER_GROUP: str = "industrial-ai-consumers"

    KAFKA_TELEMETRY_TOPIC: str = "industrial.telemetry"
    KAFKA_SAFETY_TOPIC: str = "industrial.safety"
    KAFKA_FAILURE_TOPIC: str = "industrial.failures"
    KAFKA_ALERTS_TOPIC: str = "industrial.alerts"
    KAFKA_EMERGENCY_TOPIC: str = "industrial.emergency"

    KAFKA_SECURITY_PROTOCOL: Literal[
        "PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"
    ] = "PLAINTEXT"

    # ========================================================
    # 6. MQTT
    # ========================================================

    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = Field(default=1883, ge=1, le=65535)

    MQTT_USERNAME: str = ""
    MQTT_PASSWORD: SecretStr = SecretStr("")

    MQTT_TOPIC_PREFIX: str = "industrial"
    MQTT_KEEPALIVE: int = Field(default=60, ge=1)
    MQTT_TLS_ENABLED: bool = False

    # ========================================================
    # 7. Machine Learning
    # ========================================================

    MODEL_DIR: str = "artifacts/models"

    ENABLE_ANOMALY_DETECTION: bool = True
    ENABLE_FAILURE_PREDICTION: bool = True
    ENABLE_RUL_PREDICTION: bool = True

    ANOMALY_MODEL_PATH: str = (
        "artifacts/models/anomaly/model.joblib"
    )

    FAILURE_MODEL_PATH: str = (
        "artifacts/models/failure/model.joblib"
    )

    RUL_MODEL_PATH: str = (
        "artifacts/models/rul/model.joblib"
    )

    MODEL_INFERENCE_TIMEOUT: int = Field(
        default=30, ge=1
    )

    # ========================================================
    # 8. Computer Vision / Safety
    # ========================================================

    ENABLE_SAFETY_DETECTION: bool = True

    CV_MODEL_DIR: str = "artifacts/models/cv"

    DEFECT_MODEL_PATH: str = (
        "artifacts/models/cv/defect_model.pt"
    )

    FIRE_MODEL_PATH: str = (
        "artifacts/models/cv/fire_model.pt"
    )

    SMOKE_MODEL_PATH: str = (
        "artifacts/models/cv/smoke_model.pt"
    )

    PPE_MODEL_PATH: str = (
        "artifacts/models/cv/ppe_model.pt"
    )

    CV_CONFIDENCE_THRESHOLD: float = Field(
        default=0.5, ge=0.0, le=1.0
    )

    MAX_IMAGE_SIZE_MB: int = Field(
        default=10, ge=1
    )

    # ========================================================
    # 9. Advanced RAG
    # ========================================================

    ENABLE_RAG: bool = True

    RAG_CHUNK_SIZE: int = Field(default=450, ge=50)
    RAG_CHUNK_OVERLAP: int = Field(default=75, ge=0)
    RAG_TOP_K: int = Field(default=5, ge=1)

    ENABLE_HYBRID_SEARCH: bool = True
    ENABLE_RERANKING: bool = True
    ENABLE_MULTI_QUERY: bool = True
    ENABLE_CITATIONS: bool = True

    EMBEDDING_MODEL: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    RERANKER_MODEL: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: SecretStr = SecretStr("")
    QDRANT_COLLECTION: str = "industrial_documents"

    @model_validator(mode="after")
    def validate_rag_settings(self) -> "Settings":
        if self.RAG_CHUNK_OVERLAP >= self.RAG_CHUNK_SIZE:
            raise ValueError(
                "RAG_CHUNK_OVERLAP must be smaller "
                "than RAG_CHUNK_SIZE"
            )
        return self

    # ========================================================
    # 10. LLM Providers
    # ========================================================

    LLM_PROVIDER: Literal[
        "openai", "anthropic", "ollama", "azure"
    ] = "ollama"

    OPENAI_API_KEY: SecretStr = SecretStr("")
    ANTHROPIC_API_KEY: SecretStr = SecretStr("")

    OPENAI_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"

    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_KEY: SecretStr = SecretStr("")
    AZURE_OPENAI_DEPLOYMENT: str = ""
    AZURE_OPENAI_API_VERSION: str = "2024-10-21"

    LLM_TEMPERATURE: float = Field(
        default=0.2, ge=0.0, le=2.0
    )

    LLM_MAX_TOKENS: int = Field(
        default=2048, ge=1
    )

    LLM_TIMEOUT_SECONDS: int = Field(
        default=60, ge=1
    )

    # ========================================================
    # 11. Agents
    # ========================================================

    ENABLE_AGENTIC_AI: bool = True

    ENABLE_MAINTENANCE_AGENT: bool = True
    ENABLE_SAFETY_AGENT: bool = True
    ENABLE_COST_AGENT: bool = True
    ENABLE_EMERGENCY_AGENT: bool = True
    ENABLE_INSPECTION_AGENT: bool = True

    AGENT_MAX_ITERATIONS: int = Field(
        default=10, ge=1
    )

    AGENT_TIMEOUT_SECONDS: int = Field(
        default=120, ge=1
    )

    HUMAN_APPROVAL_REQUIRED: bool = True

    # ========================================================
    # 12. Cost Intelligence
    # ========================================================

    DEFAULT_CURRENCY: str = "INR"

    ENABLE_REPAIR_COST_ANALYSIS: bool = True
    ENABLE_REPLACEMENT_COST_ANALYSIS: bool = True
    ENABLE_DOWNTIME_COST_ANALYSIS: bool = True
    ENABLE_ROI_ANALYSIS: bool = True

    # ========================================================
    # 13. Security
    # ========================================================

    SECRET_KEY: SecretStr = SecretStr("")
    JWT_SECRET_KEY: SecretStr = SecretStr("")

    JWT_ALGORITHM: Literal["HS256", "HS384", "HS512"] = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, ge=1
    )

    ENABLE_RATE_LIMITING: bool = True
    ENABLE_PII_PROTECTION: bool = True
    ENABLE_AUDIT_LOGGING: bool = True
    ENABLE_PROMPT_INJECTION_PROTECTION: bool = True

    CORS_ORIGINS: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.CORS_ORIGINS.split(",")
            if origin.strip()
        ]

    # ========================================================
    # 14. Notifications
    # ========================================================

    ENABLE_EMAIL_NOTIFICATIONS: bool = True
    ENABLE_SMS_NOTIFICATIONS: bool = False
    ENABLE_PUSH_NOTIFICATIONS: bool = True

    SMTP_HOST: str = ""
    SMTP_PORT: int = Field(default=587, ge=1, le=65535)
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: SecretStr = SecretStr("")
    SMTP_FROM_EMAIL: str = ""

    # ========================================================
    # 15. Safety and Emergency
    # ========================================================

    SAFETY_ALERTS_ENABLED: bool = True
    EMERGENCY_ALERTS_ENABLED: bool = True

    SAFETY_ALERT_THRESHOLD: float = Field(
        default=0.8, ge=0.0, le=1.0
    )

    EMERGENCY_ALERT_THRESHOLD: float = Field(
        default=0.9, ge=0.0, le=1.0
    )

    # ========================================================
    # 16. Monitoring
    # ========================================================

    ENABLE_METRICS: bool = True
    METRICS_PORT: int = Field(
        default=9090, ge=1, le=65535
    )

    ENABLE_TRACING: bool = True
    OTEL_SERVICE_NAME: str = "industrial-ai-api"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = ""

    ENABLE_MODEL_MONITORING: bool = True
    ENABLE_DATA_DRIFT_MONITORING: bool = True
    ENABLE_RAG_EVALUATION: bool = True
    ENABLE_AGENT_EVALUATION: bool = True

    # ========================================================
    # 17. AWS
    # ========================================================

    AWS_REGION: str = "ap-south-1"

    AWS_S3_TELEMETRY_BUCKET: str = ""
    AWS_S3_DATASETS_BUCKET: str = ""
    AWS_S3_MODELS_BUCKET: str = ""
    AWS_S3_DOCUMENTS_BUCKET: str = ""
    AWS_S3_REPORTS_BUCKET: str = ""
    AWS_S3_BACKUPS_BUCKET: str = ""

    # ========================================================
    # 18. Startup Validation
    # ========================================================

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        if self.APP_ENV != "production":
            return self

        for name in ("SECRET_KEY", "JWT_SECRET_KEY"):
            value = getattr(self, name).get_secret_value()

            if len(value) < 32:
                raise ValueError(
                    f"{name} must contain at least "
                    "32 characters in production"
                )

        if self.DEBUG:
            raise ValueError(
                "DEBUG must be false in production"
            )

        if self.LLM_PROVIDER == "openai":
            if not self.OPENAI_API_KEY.get_secret_value():
                raise ValueError(
                    "OPENAI_API_KEY is required"
                )

        if self.LLM_PROVIDER == "anthropic":
            if not self.ANTHROPIC_API_KEY.get_secret_value():
                raise ValueError(
                    "ANTHROPIC_API_KEY is required"
                )

        if self.LLM_PROVIDER == "azure":
            if not (
                self.AZURE_OPENAI_ENDPOINT
                and self.AZURE_OPENAI_DEPLOYMENT
                and self.AZURE_OPENAI_API_KEY.get_secret_value()
            ):
                raise ValueError(
                    "Azure OpenAI configuration is incomplete"
                )

        return self


# ============================================================
# Cached Settings
# ============================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one cached Settings instance per process."""
    return Settings()


settings = get_settings()


# ============================================================
# Local Configuration Check
# ============================================================

if __name__ == "__main__":
    print("Industrial AI Configuration")
    print("-" * 40)
    print(f"Application : {settings.APP_NAME}")
    print(f"Environment : {settings.APP_ENV}")
    print(f"API Port    : {settings.API_PORT}")
    print(f"Database    : {settings.DB_HOST}")
    print(f"Redis       : {settings.REDIS_HOST}")
    print(f"Kafka       : {settings.KAFKA_BOOTSTRAP_SERVERS}")
    print(f"MQTT        : {settings.MQTT_BROKER_HOST}")
    print(f"RAG Enabled : {settings.ENABLE_RAG}")
    print(f"LLM Provider: {settings.LLM_PROVIDER}")
    print(f"Metrics     : {settings.ENABLE_METRICS}")
    print("Configuration loaded successfully.")
