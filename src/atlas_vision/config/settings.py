import os
from typing import Literal

from pydantic import AmqpDsn, BaseModel, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    name: str = "llama3.2"
    base_url: str | None = "http://ollama:11434/v1"
    api_key: str | None = None


class Settings(BaseSettings):
    api_key: str | None = None
    celery_broker_url: RedisDsn | AmqpDsn | Literal["memory://localhost/"] = RedisDsn(
        url="redis://redis:6379/0"
    )
    redis_url: RedisDsn = RedisDsn(url="redis://redis:6379/0")
    sentry_dsn: str | None = None

    report_model: LLMConfig = LLMConfig()
    chat_model: LLMConfig = LLMConfig(
        name="ollama_chat/llama3.2",
        base_url="http://ollama:11434",
    )  # Chat model is different because it's used by smolagents instead of pydantic-ai
    formatter_model: LLMConfig = LLMConfig()

    picsellia_sdk_custom_logging: bool = False
    fast_sam_path: str = "FastSAM-x.pt"

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"),
        env_nested_delimiter="_",
        env_nested_max_split=1,
        env_parse_none_str="None",
        frozen=True,
    )


settings = Settings()  # type: ignore[call-arg]
