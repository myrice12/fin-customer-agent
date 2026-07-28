"""
统一配置中心 — 基于 pydantic-settings

集中读取和管理所有环境变量，避免散落的 os.getenv 调用与配置漂移。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM 相关配置"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="OPENAI_BASE_URL",
    )
    model_name: str = Field(default="gpt-4o-mini", alias="MODEL_NAME")
    model_temperature: float = Field(default=0.0, alias="MODEL_TEMPERATURE")


class EmbeddingSettings(BaseSettings):
    """Embedding 模型相关配置"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedding_api_base: str = Field(default="", alias="EMBEDDING_API_BASE")
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_model_name: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        alias="EMBEDDING_MODEL_NAME",
    )
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")
    embedding_cooldown_seconds: int = Field(default=60, alias="EMBEDDING_COOLDOWN_SECONDS")


class MemorySettings(BaseSettings):
    """记忆系统相关配置"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    faiss_index_path: str = Field(
        default="./vector_store/faiss_index",
        alias="FAISS_INDEX_PATH",
    )
    vector_store_type: Literal["faiss"] = Field(
        default="faiss",
        alias="VECTOR_STORE_TYPE",
    )

    short_term_max_turns: int = Field(default=20, alias="SHORT_TERM_MAX_TURNS")
    short_term_ttl_seconds: int = Field(default=1800, alias="SHORT_TERM_TTL_SECONDS")
    working_memory_max_entries: int = Field(
        default=50,
        alias="WORKING_MEMORY_MAX_ENTRIES",
    )
    cleanup_interval_seconds: int = Field(
        default=300,
        alias="CLEANUP_INTERVAL_SECONDS",
    )
    cleanup_max_age_seconds: int = Field(
        default=1800,
        alias="CLEANUP_MAX_AGE_SECONDS",
    )


class TracingSettings(BaseSettings):
    """OpenTelemetry 链路追踪相关配置"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    otel_service_name: str = Field(
        default="fin-customer-agent",
        alias="OTEL_SERVICE_NAME",
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="",
        alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )


class ServerSettings(BaseSettings):
    """Web 服务相关配置"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    cors_origins: str = Field(
        default="http://localhost:8000",
        alias="CORS_ORIGINS",
    )


class HarnessSettings(BaseSettings):
    """安全沙箱 / 认证相关配置"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_keys: str = Field(default="", alias="API_KEYS")
    sandbox_timeout_seconds: float = Field(
        default=30.0,
        alias="SANDBOX_TIMEOUT_SECONDS",
    )

    @property
    def api_key_set(self) -> set[str]:
        if not self.api_keys:
            return set()
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


class Settings(BaseSettings):
    """全局配置聚合"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    harness: HarnessSettings = Field(default_factory=HarnessSettings)

    @property
    def api_key_set(self) -> set[str]:
        return self.harness.api_key_set


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()